module L96FigureFactory

if get(ENV, "GKSwstype", "") == ""
    ENV["GKSwstype"] = "100"
end

using Plots
using Statistics

function _setup_style(dpi::Int)
    default(
        dpi=dpi,
        size=(1200, 900),
        linewidth=2,
        markerstrokewidth=0.0,
        grid=true,
        gridalpha=0.22,
        framestyle=:box,
        legend=:best,
        legendfontsize=10,
        guidefontsize=12,
        tickfontsize=10,
        titlefontsize=13,
        foreground_color_legend=:black,
        background_color_legend=:white,
    )
    return nothing
end

function read_epoch_losses(training_metrics_csv::AbstractString)
    losses = Float64[]
    epochs = Int[]
    isfile(training_metrics_csv) || error("Training metrics CSV not found: $training_metrics_csv")

    in_epoch_block = false
    for line in eachline(training_metrics_csv)
        s = strip(line)
        isempty(s) && continue
        if startswith(s, "#")
            in_epoch_block = startswith(s, "# epoch,")
            continue
        end
        in_epoch_block || continue
        parts = split(s, ",")
        length(parts) >= 2 || continue
        push!(epochs, parse(Int, strip(parts[1])))
        push!(losses, parse(Float64, strip(parts[2])))
    end
    return epochs, losses
end

function save_training_figure(training_metrics_csv::AbstractString,
                              eval_rows::Vector{Dict{String,Any}},
                              out_path::AbstractString;
                              dpi::Int=180)
    _setup_style(dpi)
    mkpath(dirname(out_path))

    epochs, losses = read_epoch_losses(training_metrics_csv)
    kl_epochs = Int[]
    kl_vals = Float64[]
    for row in eval_rows
        push!(kl_epochs, Int(row["epoch"]))
        push!(kl_vals, Float64(row["avg_mode_kl_clipped"]))
    end

    p1 = plot(epochs, losses;
              label="Train loss",
              color=:dodgerblue3,
              marker=:circle,
              markersize=3,
              xlabel="Epoch",
              ylabel="Loss",
              title="Training Loss vs Epoch",
              left_margin=12Plots.mm,
              right_margin=5Plots.mm,
              top_margin=5Plots.mm,
              bottom_margin=7Plots.mm)

    p2 = plot(kl_epochs, kl_vals;
              label="Avg mode KL",
              color=:firebrick3,
              marker=:star5,
              markersize=5,
              xlabel="Epoch",
              ylabel="KL",
              title="Langevin KL vs Epoch",
              left_margin=12Plots.mm,
              right_margin=5Plots.mm,
              top_margin=5Plots.mm,
              bottom_margin=8Plots.mm)
    hline!(p2, [0.05]; label="Target 0.05", color=:black, linestyle=:dash)

    fig = plot(
        p1, p2;
        layout=(2, 1),
        size=(1260, 1020),
        left_margin=6Plots.mm,
        right_margin=6Plots.mm,
        top_margin=5Plots.mm,
        bottom_margin=6Plots.mm,
    )
    savefig(fig, out_path)
    return out_path
end

end # module
