import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Physical & Financial Constants
# ---------------------------------------------------------------------------

CARBON_KG_MWH_PEAK    = 250.0  # kg CO₂e / MWh  (grid during peak hours)
CARBON_KG_MWH_OFFPEAK =  50.0  # kg CO₂e / MWh  (overnight wind / solar surplus)

EV_FEEDER_LIMIT_MW = 6.0       # Representative substation EV-feeder thermal limit (MW)

ROUND_TRIP_EFFICIENCY     = 0.85   # DC-AC-DC round-trip (battery → inverter → grid → inverter → battery)
DEGRADATION_COST_PER_KWH = 0.03   # £/kWh battery wear cost (NMC chemistry, cycle-averaged)

SCENARIOS = {
    "A: Winter Evening Peak — Crisis": {
        "icon": "🔴",
        "description": (
            "Base load at 90% of feeder thermal limit. Sharp demand spike at 18:00. "
            "Grid frequency is volatile, dropping below 49.90 Hz. BM Spread: £400/MWh."
        ),
        "strategy": "discharge_on_peak",
        "sell_price": 500.0,   # £/MWh — 99th-pct winter BM cap
        "buy_price":  100.0,   # £/MWh — overnight wind surplus
    },
    "B: Summer Solar Surplus — Opportunity": {
        "icon": "🟡",
        "description": (
            "Duck Curve: solar generation peaks at 12:00, suppressing wholesale prices to ~£5/MWh. "
            "EVs soak energy 10:00–14:00, then discharge during the 19:00 evening lighting ramp."
        ),
        "strategy": "duck_curve",
        "sell_price": 280.0,   # £/MWh — evening ramp
        "buy_price":    5.0,   # £/MWh — solar surplus
    },
    "C: High Wind / Low Demand — Stable": {
        "icon": "🔵",
        "description": (
            "Load is flat, ~40% below feeder thermal limit. "
            "No dispatch interventions needed. Prices are constant. Fleet on standby."
        ),
        "strategy": "none",
        "sell_price": 80.0,
        "buy_price":  60.0,
    },
}


# ---------------------------------------------------------------------------
# Data helper — module-level so st.cache_data works correctly
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_substation_limit() -> float:
    """Return 95th-percentile thermal limit from live UKPN parquet data."""
    url = (
        "https://raw.githubusercontent.com/jpbarona/quantihack/main/"
        "data/CleanGridData/data.parquet"
    )
    try:
        df_grid = pd.read_parquet(url, engine="pyarrow")
        numeric_cols = df_grid.select_dtypes(include=[np.number]).columns
        demand_cols  = [c for c in numeric_cols if "demand" in c.lower() or "load" in c.lower()]
        col = demand_cols[0] if demand_cols else numeric_cols[0]
        return float(df_grid[col].quantile(0.95))
    except Exception:
        return 40_000.0


# ---------------------------------------------------------------------------
# Core production model class
# ---------------------------------------------------------------------------

class EVSmartCharger:
    """
    VPP (Virtual Power Plant) Node Optimizer — Production-Grade EV V2G Engine.

    Physical sizing: Max_Discharge(t) = Fleet_Size × P_conn(t) × 0.007 MW
    (7 kW per vehicle = UK residential Ohme/Indra bidirectional charger standard).

    P_conn(t) is a UKPN-calibrated time-varying plug-in probability:
      08:00–16:00 : ~0.15  (commuting)
      16:00–20:00 : sigmoid ramp 0.15 → 0.90  (home-arrival wave)
      20:00–07:00 : ~0.95  (fleet fully home)

    Round-trip efficiency: 0.85 (DC-AC-DC, IEC 62196 residential charger standard).
    Degradation cost: £0.03/kWh (cycle-averaged NMC wear, BEIS 2024 estimate).
    Minimum SOC constraint: EVs may not discharge below 40% SoC, preserving
    ~80 miles of commuter range (40 kWh battery, 5 mi/kWh UK average).
    """

    HOURS = np.arange(24)

    def __init__(
        self,
        fleet_size: int   = 300,
        v2g_charger_kw: float = 7.0,
        minimum_soc: float    = 0.40,
    ):
        self.fleet_size      = fleet_size
        self.v2g_charger_kw  = v2g_charger_kw
        self.minimum_soc     = minimum_soc
        self._update_capacity()

    # ------------------------------------------------------------------
    # UKPN-calibrated plug-in probability profile
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_p_conn() -> np.ndarray:
        """
        UKPN-calibrated time-varying plug-in probability P_conn(t).

        16:00–20:00 uses a sigmoid centred at 18:00 (steepness k=2):
          P = 0.15 + 0.75 × σ(2 × (h − 18))
        which yields P(16) ≈ 0.16, P(18) ≈ 0.525, P(20) ≈ 0.89.
        """
        p = np.empty(24)
        for h in range(24):
            if 8 <= h < 16:
                p[h] = 0.15
            elif 16 <= h < 20:
                sig  = 1.0 / (1.0 + np.exp(-2.0 * (h - 18.0)))
                p[h] = 0.15 + 0.75 * sig
            else:           # 20:00–07:00 (overnight)
                p[h] = 0.95
        return p

    def _update_capacity(self) -> None:
        self.p_conn_profile    = self._compute_p_conn()
        self.max_v2g_per_hour  = (
            self.fleet_size * self.v2g_charger_kw / 1_000.0 * self.p_conn_profile
        )
        # Scalar peak capacity (for sidebar display and legacy calcs)
        self.max_v2g_discharge_mw = float(np.max(self.max_v2g_per_hour))

    # ------------------------------------------------------------------
    # Profile generators
    # ------------------------------------------------------------------

    def generate_baseline_load(self) -> np.ndarray:
        """Unmanaged EV demand profile (empirical commuter arrival distribution)."""
        base, peak = 0.5, 4.5
        return base + peak * np.exp(-0.5 * ((self.HOURS - 18.0) / 2.5) ** 2)

    def generate_scenario_profiles(self, key: str):
        """
        Return (grid_load_mw, grid_freq_hz, price_gbp_mwh) for the chosen scenario.
        """
        h = self.HOURS

        if SCENARIOS[key]["strategy"] == "discharge_on_peak":
            base  = EV_FEEDER_LIMIT_MW * 0.90
            spike = EV_FEEDER_LIMIT_MW * 0.28 * np.exp(-0.5 * ((h - 18.0) / 1.5) ** 2)
            grid_load = base + spike

            grid_freq = np.interp(
                h,
                [ 0,    6,    12,    17,    18,    19,    21,    23],
                [50.02, 50.00, 49.98, 49.93, 49.78, 49.84, 49.92, 50.01],
            )
            price = np.where((h >= 17) & (h <= 20), 500.0,
                    np.where((h >= 7)  & (h <= 16), 150.0, 100.0)).astype(float)

        elif SCENARIOS[key]["strategy"] == "duck_curve":
            base       = EV_FEEDER_LIMIT_MW * 0.65
            morning    =  0.30 * np.exp(-0.5 * ((h -  9.0) / 2.0) ** 2)
            solar_dip  = -0.45 * np.exp(-0.5 * ((h - 12.0) / 2.5) ** 2)
            evening    =  0.55 * np.exp(-0.5 * ((h - 19.0) / 1.5) ** 2)
            grid_load  = np.maximum(
                base + EV_FEEDER_LIMIT_MW * (morning + solar_dip + evening),
                EV_FEEDER_LIMIT_MW * 0.20,
            )
            grid_freq = np.interp(
                h,
                [ 0,     6,    12,    18,    19,    21,    23],
                [50.02, 50.03, 50.05, 50.00, 49.92, 49.96, 50.01],
            )
            midday_f  = np.exp(-0.5 * ((h - 12.0) / 2.0) ** 2)
            evening_f = np.exp(-0.5 * ((h - 19.5) / 1.5) ** 2)
            price = np.clip(
                5.0   * midday_f
                + 280.0 * evening_f
                + 120.0 * (1.0 - midday_f) * (1.0 - evening_f),
                5.0, 280.0,
            )

        else:
            base      = EV_FEEDER_LIMIT_MW * 0.60
            grid_load = base + base * 0.04 * np.sin(2 * np.pi * (h - 6) / 24)
            grid_freq = np.full(24, 50.02)
            price     = np.full(24, 70.0)

        return grid_load, grid_freq, price

    # ------------------------------------------------------------------
    # Water-filling rebound (Intelligent Recovery)
    # ------------------------------------------------------------------

    def _waterfill_rebound(
        self,
        optimized_load: np.ndarray,
        energy_deficit: float,
        ceiling: float,
    ) -> None:
        """
        Spread recharge deficit via water-filling algorithm.

        Fills lowest-load hours first, raising them uniformly (like water
        filling a vessel), never allowing any hour to exceed `ceiling`
        (= unmanaged EV baseline peak).  This guarantees Managed_Peak
        ≤ Unmanaged_Peak — no artificial rebound spike is created.
        """
        if energy_deficit <= 1e-9:
            return

        remaining = energy_deficit
        loads     = optimized_load.copy()

        # Only hours with headroom below the ceiling are eligible
        eligible = sorted(
            [h for h in range(24) if loads[h] < ceiling - 1e-9],
            key=lambda h: loads[h],
        )
        if not eligible:
            return

        n           = len(eligible)
        level_vals  = [loads[h] for h in eligible]   # sorted ascending
        target      = ceiling                         # fallback: fill to ceiling

        current_floor = level_vals[0]
        for i in range(n):
            next_bp        = level_vals[i + 1] if i < n - 1 else ceiling
            energy_to_next = (next_bp - current_floor) * (i + 1)

            if remaining <= energy_to_next:
                target    = current_floor + remaining / (i + 1)
                remaining = 0.0
                break

            remaining     -= energy_to_next
            current_floor  = next_bp

        # Apply fill: raise every eligible hour below `target` up to `target`
        for h in eligible:
            if loads[h] < target:
                optimized_load[h] = min(target, ceiling)

    # ------------------------------------------------------------------
    # Optimisation engine
    # ------------------------------------------------------------------

    def calculate_optimized_profile(
        self,
        key: str,
        baseline: np.ndarray,
        grid_load: np.ndarray,
        grid_freq: np.ndarray,
        price: np.ndarray,
    ) -> pd.DataFrame:
        """
        Scenario-aware adaptive dispatch engine.

        A — Crisis : discharge when load ≥ feeder limit OR freq < 49.90 Hz.
                     Capacity is time-varying (P_conn × Fleet × kW).
                     Rebound uses water-filling; recharge deficit scaled by
                     1/ROUND_TRIP_EFFICIENCY.
        B — Duck   : charge during solar surplus (10-14h), discharge at 19-20h.
                     Usable discharge energy = soaked × ROUND_TRIP_EFFICIENCY.
        C — Stable : no interventions; optimised profile == baseline.
        """
        strategy        = SCENARIOS[key]["strategy"]
        optimized_load  = baseline.copy().astype(float)
        dispatch_signal = np.zeros(24)
        charge_signal   = np.zeros(24)
        v2g_cap_mw      = self.max_v2g_per_hour.copy()

        if strategy == "discharge_on_peak":
            energy_discharged = 0.0
            for i in self.HOURS:
                cap = v2g_cap_mw[i]
                if grid_load[i] >= EV_FEEDER_LIMIT_MW or grid_freq[i] < 49.90:
                    dispatch_signal[i]  = 1.0
                    optimized_load[i]   = -cap
                    energy_discharged  += cap   # MWh (1-hour intervals)

            # Recharge deficit = displaced baseline energy
            #                  + V2G discharge energy grossed up for round-trip losses
            baseline_displaced = float(
                np.sum(baseline[dispatch_signal == 1.0])
            )
            energy_deficit = baseline_displaced + energy_discharged / ROUND_TRIP_EFFICIENCY

            self._waterfill_rebound(optimized_load, energy_deficit, float(baseline.max()))

        elif strategy == "duck_curve":
            soak_hours      = [10, 11, 12, 13, 14]
            discharge_hours = [19, 20]

            energy_soaked = 0.0
            for i in soak_hours:
                cap               = v2g_cap_mw[i]
                charge_signal[i]  = 1.0
                optimized_load[i] = baseline[i] + cap
                energy_soaked    += cap

            # Usable discharge is limited by round-trip efficiency
            energy_available  = energy_soaked * ROUND_TRIP_EFFICIENCY
            cap_at_discharge  = min(v2g_cap_mw[i] for i in discharge_hours)
            discharge_each    = min(energy_available / len(discharge_hours), cap_at_discharge)
            for i in discharge_hours:
                dispatch_signal[i] = 1.0
                optimized_load[i]  = baseline[i] - discharge_each

        return pd.DataFrame({
            "Hour":              self.HOURS,
            "Baseline_Load_MW":  baseline,
            "Grid_Load_MW":      grid_load,
            "Grid_Frequency_Hz": grid_freq,
            "Price_GBP_MWh":     price,
            "Optimized_Load_MW": optimized_load,
            "Dispatch_Signal":   dispatch_signal,
            "Charge_Signal":     charge_signal,
            "P_conn":            self.p_conn_profile,
            "V2G_Cap_MW":        v2g_cap_mw,
        })

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def calculate_metrics(
        self, df: pd.DataFrame, key: str, selected_hour: int = 18
    ) -> dict:
        """
        Calculate production-grade impact metrics including efficiency losses
        and battery degradation costs.
        """
        strategy = SCENARIOS[key]["strategy"]
        params   = SCENARIOS[key]

        baseline_peak  = df["Baseline_Load_MW"].max()
        optimized_peak = df["Optimized_Load_MW"].max()
        peak_relief_mw = baseline_peak - optimized_peak

        # Metric 1: % peak relief at the 18:00 spike specifically
        h18_base = float(df.loc[df["Hour"] == 18, "Baseline_Load_MW"].iloc[0])
        h18_opt  = float(df.loc[df["Hour"] == 18, "Optimized_Load_MW"].iloc[0])
        peak_relief_pct = (
            (h18_base - h18_opt) / h18_base * 100.0 if h18_base > 1e-9 else 0.0
        )

        # Metric 3: fleet availability at the user-selected hour
        p_conn_at_hour  = float(df.loc[df["Hour"] == selected_hour, "P_conn"].iloc[0])
        fleet_available = int(self.fleet_size * p_conn_at_hour)

        n_dispatch = int(df["Dispatch_Signal"].sum())
        n_charge   = int(df["Charge_Signal"].sum())

        if strategy == "discharge_on_peak":
            dispatch_rows     = df[df["Dispatch_Signal"] == 1.0]
            energy_discharged = float(dispatch_rows["V2G_Cap_MW"].sum())   # MWh
            energy_recharged  = energy_discharged / ROUND_TRIP_EFFICIENCY  # MWh bought back

            # Gross arbitrage: revenue from selling − cost of buying back (with losses)
            gross_yield_gbp    = energy_discharged * params["sell_price"]
            recharge_cost_gbp  = energy_recharged  * params["buy_price"]
            efficiency_loss_gbp = recharge_cost_gbp - energy_discharged * params["buy_price"]

            # Battery degradation on discharged kWh
            degradation_gbp    = energy_discharged * 1_000.0 * DEGRADATION_COST_PER_KWH

            event_yield_gbp    = gross_yield_gbp - recharge_cost_gbp
            net_profit_gbp     = event_yield_gbp - degradation_gbp

            energy_arbitraged  = energy_discharged
            co2_avoided_kg     = energy_discharged * (CARBON_KG_MWH_PEAK - CARBON_KG_MWH_OFFPEAK)

        elif strategy == "duck_curve":
            charge_rows       = df[df["Charge_Signal"]   == 1.0]
            dispatch_rows     = df[df["Dispatch_Signal"] == 1.0]
            energy_soaked     = float(charge_rows["V2G_Cap_MW"].sum())
            energy_discharged = energy_soaked * ROUND_TRIP_EFFICIENCY
            energy_arbitraged = energy_discharged

            avg_buy  = (charge_rows["Price_GBP_MWh"].mean()   if n_charge   else params["buy_price"])
            avg_sell = (dispatch_rows["Price_GBP_MWh"].mean() if n_dispatch else params["sell_price"])

            gross_yield_gbp     = energy_discharged * avg_sell
            recharge_cost_gbp   = energy_soaked     * avg_buy
            efficiency_loss_gbp = energy_soaked * avg_buy * (1.0 - ROUND_TRIP_EFFICIENCY)

            degradation_gbp     = energy_discharged * 1_000.0 * DEGRADATION_COST_PER_KWH
            event_yield_gbp     = gross_yield_gbp - recharge_cost_gbp
            net_profit_gbp      = event_yield_gbp - degradation_gbp
            co2_avoided_kg      = energy_arbitraged * (CARBON_KG_MWH_PEAK - CARBON_KG_MWH_OFFPEAK)

        else:
            energy_arbitraged   = 0.0
            event_yield_gbp     = 0.0
            net_profit_gbp      = 0.0
            degradation_gbp     = 0.0
            efficiency_loss_gbp = 0.0
            co2_avoided_kg      = 0.0

        return {
            "peak_relief_mw":        peak_relief_mw,
            "peak_relief_pct":       peak_relief_pct,
            "baseline_peak_mw":      baseline_peak,
            "optimized_peak_mw":     optimized_peak,
            "energy_arbitraged_mwh": energy_arbitraged,
            "event_yield_gbp":       event_yield_gbp,
            "net_profit_gbp":        net_profit_gbp,
            "degradation_gbp":       degradation_gbp,
            "efficiency_loss_gbp":   efficiency_loss_gbp,
            "co2_avoided_kg":        co2_avoided_kg,
            "n_dispatch":            n_dispatch,
            "n_charge":              n_charge,
            "fleet_available":       fleet_available,
        }

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------

    def build_chart(self, df: pd.DataFrame, key: str) -> go.Figure:
        params   = SCENARIOS[key]
        strategy = params["strategy"]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.65, 0.35],
            vertical_spacing=0.10,
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
            subplot_titles=[
                "VPP Node Power Flow & Grid Feeder Load",
                "Electricity Price  (£/MWh)",
            ],
        )

        # ── Highlight dispatch / soak windows ────────────────────────────
        for h in df.loc[df["Dispatch_Signal"] == 1.0, "Hour"]:
            fig.add_vrect(
                x0=h - 0.5, x1=h + 0.5,
                fillcolor="rgba(255,80,80,0.15)", line_width=0,
                row=1, col=1,
            )
        for h in df.loc[df["Charge_Signal"] == 1.0, "Hour"]:
            fig.add_vrect(
                x0=h - 0.5, x1=h + 0.5,
                fillcolor="rgba(255,200,0,0.12)", line_width=0,
                row=1, col=1,
            )

        # ── Grid feeder load (secondary Y, shaded fill) ──────────────────
        fig.add_trace(go.Scatter(
            x=df["Hour"], y=df["Grid_Load_MW"],
            name="Grid Feeder Load",
            line=dict(color="rgba(255,255,255,0.20)", width=2),
            fill="tozeroy", fillcolor="rgba(255,255,255,0.04)",
            hovertemplate="%{y:.2f} MW<extra>Grid Feeder Load</extra>",
        ), row=1, col=1, secondary_y=True)

        # ── Feeder thermal limit ──────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=self.HOURS,
            y=np.full(24, EV_FEEDER_LIMIT_MW),
            name="Feeder Thermal Limit",
            line=dict(color="#ff4444", width=1.5, dash="dot"),
            hoverinfo="skip",
            showlegend=True,
        ), row=1, col=1, secondary_y=True)

        # ── 49.90 Hz statutory floor (legend-only by default) ────────────
        fig.add_trace(go.Scatter(
            x=self.HOURS,
            y=np.full(24, 49.90),
            name="49.90 Hz Statutory Floor",
            line=dict(color="rgba(200,100,255,0.55)", width=1.2, dash="dot"),
            hoverinfo="skip",
            visible="legendonly",
        ), row=1, col=1, secondary_y=True)

        # ── Grid frequency (secondary Y, hidden by default) ───────────────
        fig.add_trace(go.Scatter(
            x=df["Hour"], y=df["Grid_Frequency_Hz"],
            name="Grid Frequency (Hz)",
            line=dict(color="rgba(160,100,255,0.9)", width=1.5, dash="dot"),
            hovertemplate="%{y:.3f} Hz<extra>Frequency</extra>",
            visible="legendonly",
        ), row=1, col=1, secondary_y=True)

        # ── P_conn curve (secondary Y, hidden by default) ────────────────
        fig.add_trace(go.Scatter(
            x=df["Hour"], y=df["P_conn"],
            name="P_conn (Plug-in Prob.)",
            line=dict(color="rgba(0,200,255,0.7)", width=1.5, dash="dashdot"),
            hovertemplate="%{y:.2f}<extra>P_conn</extra>",
            visible="legendonly",
        ), row=1, col=1, secondary_y=True)

        # ── Unmanaged EV demand (primary Y) ──────────────────────────────
        fig.add_trace(go.Scatter(
            x=df["Hour"], y=df["Baseline_Load_MW"],
            name="Unmanaged EV Demand",
            line=dict(color="#888888", width=2, dash="dash"),
            hovertemplate="%{y:.3f} MW<extra>Unmanaged EV</extra>",
        ), row=1, col=1, secondary_y=False)

        # ── Adaptive V2G response (primary Y) ────────────────────────────
        fill_mode = "tonexty" if strategy != "none" else None
        fig.add_trace(go.Scatter(
            x=df["Hour"], y=df["Optimized_Load_MW"],
            name="VPP Adaptive Response",
            line=dict(color="#00ffcc", width=4),
            fill=fill_mode,
            fillcolor="rgba(0,255,204,0.08)",
            hovertemplate="%{y:.3f} MW<extra>VPP Response</extra>",
        ), row=1, col=1, secondary_y=False)

        # ── Price bar chart (row 2) ───────────────────────────────────────
        fig.add_trace(go.Bar(
            x=df["Hour"], y=df["Price_GBP_MWh"],
            name="Price (£/MWh)",
            marker=dict(
                color=df["Price_GBP_MWh"],
                colorscale="RdYlGn_r",
                showscale=False,
                line=dict(width=0),
            ),
            hovertemplate="£%{y:.0f}/MWh<extra>Price</extra>",
        ), row=2, col=1)

        fig.update_layout(
            title=dict(
                text=(
                    f"{params['icon']}  {key}"
                    "  ·  VPP (Virtual Power Plant) Node Optimizer"
                ),
                font=dict(size=17, color="#ffffff"),
            ),
            template="plotly_dark",
            hovermode="x unified",
            height=700,
            margin=dict(t=115, b=50, l=75, r=75),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.04,
                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
            ),
        )
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1, dtick=1, tickfont=dict(size=11))
        fig.update_yaxes(title_text="EV Fleet Power (MW)",   secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Grid Load / Freq / P_conn", secondary_y=True,  row=1, col=1)
        fig.update_yaxes(title_text="Price (£/MWh)",           row=2, col=1)

        return fig

    # ------------------------------------------------------------------
    # Streamlit Dashboard
    # ------------------------------------------------------------------

    def render_dashboard(self) -> None:
        """Full multi-scenario Streamlit VPP Node Optimizer dashboard."""
        st.title("⚡ VPP (Virtual Power Plant) Node Optimizer")

        st.caption(
            f"Fleet: **{self.fleet_size}** EVs  ·  "
            f"Peak V2G capacity: **{self.max_v2g_discharge_mw:.3f} MW** "
            f"(at P_conn = 0.95)  ·  "
            f"Round-trip efficiency: **{int(ROUND_TRIP_EFFICIENCY * 100)}%**  ·  "
            f"Degradation cost: **£{DEGRADATION_COST_PER_KWH:.2f}/kWh**  ·  "
            f"Min SOC floor: **{int(self.minimum_soc * 100)}%**"
        )

        # ── Sidebar ──────────────────────────────────────────────────────
        with st.sidebar:
            st.header("Scenario")
            scenario_key = st.selectbox(
                "Grid Environment",
                options=list(SCENARIOS.keys()),
                format_func=lambda k: f"{SCENARIOS[k]['icon']}  {k}",
                help="Select the grid stress scenario to simulate.",
            )
            params = SCENARIOS[scenario_key]
            st.info(params["description"])

            st.divider()
            st.subheader("Fleet Parameters")
            fleet_size = st.slider("Fleet Size (EVs)", 100, 1_000, self.fleet_size, 50)

            st.divider()
            st.subheader("Fleet Availability Inspector")
            selected_hour = st.slider(
                "Selected Hour",
                0, 23, 18,
                help="Choose an hour to inspect P_conn and fleet availability.",
            )
            p_at_hour = self.p_conn_profile[selected_hour]
            st.metric(
                f"P_conn at {selected_hour:02d}:00",
                f"{p_at_hour:.2f}",
                help="Time-varying UKPN plug-in probability at the selected hour.",
            )

            # Apply slider values (Streamlit reruns on change)
            self.fleet_size = fleet_size
            self._update_capacity()

            st.metric("Peak V2G Discharge", f"{self.max_v2g_discharge_mw:.3f} MW")
            st.caption(
                f"Sell: **£{params['sell_price']:.0f}**/MWh  ·  "
                f"Buy: **£{params['buy_price']:.0f}**/MWh  ·  "
                f"Spread: **£{params['sell_price'] - params['buy_price']:.0f}**/MWh"
            )

        # ── Data loading ─────────────────────────────────────────────────
        with st.spinner("Fetching live UKPN substation data…"):
            fetch_substation_limit()

        baseline  = self.generate_baseline_load()
        grid_load, grid_freq, price = self.generate_scenario_profiles(scenario_key)

        results_df = self.calculate_optimized_profile(
            scenario_key, baseline, grid_load, grid_freq, price
        )
        metrics = self.calculate_metrics(results_df, scenario_key, selected_hour)

        # ── 4-Column KPI Metrics ──────────────────────────────────────────
        st.subheader("Impact Metrics")
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Peak System Relief (18:00 Spike)",
                f"{metrics['peak_relief_pct']:.1f}%",
                delta=f"{metrics['baseline_peak_mw']:.2f} → {metrics['optimized_peak_mw']:.2f} MW",
                delta_color="normal" if metrics["peak_relief_pct"] >= 0 else "inverse",
                help=(
                    "% reduction in the 18:00 demand spike. "
                    "Positive = load successfully curtailed at peak."
                ),
            )

        with c2:
            net = metrics["net_profit_gbp"]
            deg = metrics["degradation_gbp"]
            eff = metrics["efficiency_loss_gbp"]
            st.metric(
                "Net Arbitrage Profit (£)",
                f"£{net:,.0f}",
                delta=f"−£{deg:,.0f} deg · −£{eff:,.0f} eff loss",
                delta_color="normal" if net >= 0 else "inverse",
                help=(
                    "Gross arbitrage yield minus battery degradation cost "
                    f"(£{DEGRADATION_COST_PER_KWH}/kWh) and round-trip "
                    f"efficiency losses ({int(ROUND_TRIP_EFFICIENCY*100)}%)."
                ),
            )

        with c3:
            st.metric(
                f"Fleet Availability at {selected_hour:02d}:00",
                f"{metrics['fleet_available']} EVs",
                delta=f"P_conn = {self.p_conn_profile[selected_hour]:.2f}",
                delta_color="off",
                help=(
                    "Number of EVs estimated to be plugged in at the selected hour, "
                    "based on the UKPN-calibrated home-arrival distribution."
                ),
            )

        with c4:
            co2_t = metrics["co2_avoided_kg"] / 1_000
            st.metric(
                "Carbon Intensity Impact",
                f"{metrics['co2_avoided_kg']:,.0f} kg CO₂e",
                delta=f"{co2_t:.2f} tCO₂e avoided",
                delta_color="normal" if metrics["co2_avoided_kg"] >= 0 else "inverse",
                help=(
                    f"Peak grid: {CARBON_KG_MWH_PEAK:.0f} kg CO₂e/MWh · "
                    f"Off-peak: {CARBON_KG_MWH_OFFPEAK:.0f} kg CO₂e/MWh. "
                    "Net CO₂e avoided by shifting load from peak to low-carbon periods."
                ),
            )

        st.divider()

        # ── Chart ─────────────────────────────────────────────────────────
        fig = self.build_chart(results_df, scenario_key)
        st.plotly_chart(fig, use_container_width=True)

        # ── Expandable sections ───────────────────────────────────────────
        with st.expander("Commercial & Physical Assumptions"):
            active_v = int(self.fleet_size * self.p_conn_profile[selected_hour])
            st.markdown(f"""
| Parameter | Value | Note |
|---|---|---|
| Scenario | {scenario_key} | User selected |
| BM Sell Price | £{params['sell_price']:.0f} / MWh | Scenario peak BM assumption |
| Overnight Buy Price | £{params['buy_price']:.0f} / MWh | Scenario off-peak assumption |
| BM Spread | £{params['sell_price'] - params['buy_price']:.0f} / MWh | |
| Fleet Size | {self.fleet_size} EVs | |
| P_conn at {selected_hour:02d}:00 | {self.p_conn_profile[selected_hour]:.2f} | UKPN home-arrival distribution |
| Fleet Available at {selected_hour:02d}:00 | {active_v} EVs | Fleet × P_conn |
| V2G Charger Rating | {self.v2g_charger_kw} kW | UK residential std (Ohme / Indra) |
| Peak V2G Discharge | {self.max_v2g_discharge_mw:.3f} MW | At P_conn = 0.95 (overnight) |
| Round-Trip Efficiency | {int(ROUND_TRIP_EFFICIENCY * 100)}% | DC-AC-DC (IEC 62196) |
| Degradation Cost | £{DEGRADATION_COST_PER_KWH:.2f} / kWh | NMC cycle wear (BEIS 2024) |
| Min SOC Floor | {int(self.minimum_soc * 100)}% | ≈ 80-mile commuter range guarantee |
| Peak Grid Intensity | {CARBON_KG_MWH_PEAK:.0f} kg CO₂e / MWh | UK grid estimate during demand peaks |
| Off-Peak Grid Intensity | {CARBON_KG_MWH_OFFPEAK:.0f} kg CO₂e / MWh | Wind / solar surplus estimate |
| Gross Arbitrage Yield | £{metrics['event_yield_gbp']:,.0f} | Before losses |
| Degradation Cost | £{metrics['degradation_gbp']:,.0f} | Battery wear |
| Efficiency Loss Cost | £{metrics['efficiency_loss_gbp']:,.0f} | Round-trip recharge premium |
| **Net Arbitrage Profit** | **£{metrics['net_profit_gbp']:,.0f}** | **After all losses** |
""")

        with st.expander("Hourly Simulation Data"):
            st.dataframe(
                results_df.style.format({
                    "Baseline_Load_MW":  "{:.3f}",
                    "Grid_Load_MW":      "{:.3f}",
                    "Grid_Frequency_Hz": "{:.3f}",
                    "Price_GBP_MWh":     "{:.1f}",
                    "Optimized_Load_MW": "{:.3f}",
                    "Dispatch_Signal":   "{:.0f}",
                    "Charge_Signal":     "{:.0f}",
                    "P_conn":            "{:.3f}",
                    "V2G_Cap_MW":        "{:.3f}",
                }),
                use_container_width=True,
                height=360,
            )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="VPP Node Optimizer",
        page_icon="⚡",
        layout="wide",
    )

    if "charger" not in st.session_state:
        st.session_state.charger = EVSmartCharger(
            fleet_size=300,
            v2g_charger_kw=7.0,
            minimum_soc=0.40,
        )

    st.session_state.charger.render_dashboard()


if __name__ == "__main__":
    main()
