using Plots

"""
    plot_langevin_vs_observed(trajectory, dataset; dt_sim=1.0, dt_obs=1.0, max_lag=nothing)

Create a two-panel figure comparing a Langevin-generated trajectory against
observed data from a `NormalizedDataset`.

- Top panel: time series of the first spatial mode `u(1,t)` for observed and
  simulated data, plotted over the common time window.
- Bottom panel: autocorrelation functions (ACFs) computed via
  [`average_mode_acf`] using ALL available points from each dataset independently.

Arguments:
- `trajectory::Array{<:Real,3}`: simulated tensor with layout `(L, C, T_sim)`.
- `dataset::NormalizedDataset`: observed data with layout `(L, C, T_obs)`.

Keyword arguments:
- `dt_sim::Real`: time step for the simulated trajectory (default: 1.0).
- `dt_obs::Real`: time step for the observed data (default: 1.0).
- `max_lag::Union{Nothing,Int}`: maximum lag for the ACFs. When `nothing`,
  uses `min(T_sim, T_obs) - 1`.

Returns:
- A `Plots.Plot` object with a 2×1 layout.
"""
function plot_langevin_vs_observed(trajectory::Array{<:Real,3},
                                   dataset::NormalizedDataset;
                                   dt_sim::Real = 1.0,
                                   dt_obs::Real = 1.0,
                                   max_lag::Union{Nothing,Int}=nothing)
    L_sim, C_sim, T_sim = size(trajectory)
    L_obs, C_obs, T_obs = size(dataset.data)

    L_sim == L_obs || error("Simulated and observed data must have the same spatial length")
    C_sim == C_obs || error("Simulated and observed data must have the same number of channels")

    # Time-series comparison for u(1,t) - use common time window
    T_common = min(T_sim, T_obs)
    T_common > 1 || error("Not enough samples for time-series comparison")

    sim_series = vec(trajectory[1, 1, 1:T_common])
    obs_series = vec(dataset.data[1, 1, 1:T_common])
    
    # Create time axes
    t_sim = collect(0:T_common-1) .* dt_sim
    t_obs = collect(0:T_common-1) .* dt_obs

    traj_plot = plot(t_obs, obs_series;
                     xlabel = "Time",
                     ylabel = "x[1]",
                     title = "Time Series Comparison (Variable 1)",
                     label = "Observed (dt=$(dt_obs))",
                     color = :navy,
                     linewidth = 2)
    plot!(traj_plot, t_sim, sim_series;
          label = "Simulated (dt=$(dt_sim))",
          color = :firebrick,
          linewidth = 2,
          linestyle = :dash)

    # ACF comparison using ALL available points from each dataset independently
    sim_matrix = reshape(trajectory, L_sim * C_sim, T_sim)
    obs_matrix = reshape(dataset.data, L_obs * C_obs, T_obs)

    T_sim > 1 || error("Not enough simulated samples to compute autocorrelation")
    T_obs > 1 || error("Not enough observed samples to compute autocorrelation")

    # Determine max_lag based on the shorter dataset, but compute ACFs using full data
    T_min = min(T_sim, T_obs)
    max_lag_val = max_lag === nothing ? (T_min - 1) : min(max_lag, T_sim - 1, T_obs - 1)
    max_lag_val >= 0 || error("max_lag must be non-negative and less than both dataset lengths")

    # Use ALL points from each dataset
    acf_sim = average_mode_acf(sim_matrix, max_lag_val)
    acf_obs = average_mode_acf(obs_matrix, max_lag_val)
    
    # Create lag time axes
    lags_sim = collect(0:max_lag_val) .* dt_sim
    lags_obs = collect(0:max_lag_val) .* dt_obs

    acf_plot = plot(lags_obs, acf_obs;
                    xlabel = "Time",
                    ylabel = "ACF",
                    title = "ACF Comparison (Observed: $(T_obs) pts, Simulated: $(T_sim) pts)",
                    label = "Observed",
                    color = :navy,
                    linewidth = 2)
    plot!(acf_plot, lags_sim, acf_sim;
          label = "Simulated",
          color = :firebrick,
          linewidth = 2,
          linestyle = :dash)

    return plot(traj_plot, acf_plot; layout = (2, 1), size = (800, 600))
end

