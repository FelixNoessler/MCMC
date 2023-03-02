using Distributions
using CairoMakie
using LinearAlgebra
import StatsBase

own_theme = Theme(
    fontsize=18,
    Axis=(xgridvisible=false, ygridvisible=false,
          topspinevisible=false, rightspinevisible=false),
)
set_theme!(own_theme)

let 

    σ1, σ2 = 1, 1
    ρ = 0.8
    Σ = [σ1^2 ρ*σ1*σ2; ρ*σ1*σ2 σ2^2 ]
    μ = [0, 0]
    d = MultivariateNormal(μ, Hermitian(Σ))

    normal_samples = rand(d, 20000)
    
    norm_margin1 = Normal(μ[1], σ1)
    norm_margin2 = Normal(μ[2], σ2)

    unif_1 = cdf.(norm_margin1, normal_samples[1, :])
    unif_2 = cdf.(norm_margin2, normal_samples[2, :])

    begin
        fig = Figure()
        Axis(fig[1,1])
        hist!(unif_1; bins=50)
    
        Axis(fig[1,2])
        hist!(unif_2; bins=50)
    
        fig
    end

     

    ############ first example: Beta and Gamma
    gamma_dist = Gamma(2, 5)
    gamma_samples = quantile(gamma_dist, unif_1)

    beta_dist = Beta(5, 10)
    beta_samples = quantile(beta_dist, unif_2)


    begin
        fig = Figure()
        Axis(fig[1,1])
        density!(gamma_samples;
            color=(:steelblue3, 0.8))
        xlims!(-5, 65)

        Axis(fig[2,1])
        scatter!(gamma_samples, beta_samples;
            color=(:orange, 0.2))
        limits!(-5, 65, -0.1, 1.1)

        Axis(fig[2,2];)
        density!(beta_samples; 
            direction = :y,
            npoints = 1000,
            color=(:steelblue3, 0.8))
        ylims!(-0.1, 1.1)
        
        rowgap!(fig.layout, 1, 0)
        colgap!(fig.layout, 1, 0)

        rowsize!(fig.layout, 2, Relative(0.7))
        colsize!(fig.layout, 1, Relative(0.7))

        display(fig)
    end



    ############ first example: Categorical and LogNormal
    categorical_dist = Categorical([0.2, 0.1, 0.1, 0.4, 0.2])
    categorical_samples = quantile(categorical_dist, unif_1)
    discrete_counts = StatsBase.counts(categorical_samples, 1:5)
    
    lognormal_dist = LogNormal(2, 0.5)
    lnormal_samples = quantile(lognormal_dist, unif_2)


    begin
        fig = Figure()
        Axis(fig[1,1])
        barplot!(1:5, discrete_counts;
            color=(:steelblue3, 0.8))
        xlims!(0.5, 5.5)

        Axis(fig[2,1])
        jitter = (rand(length(categorical_samples)) .- 0.5) .* 0.8
        scatter!(categorical_samples .+ jitter, lnormal_samples;
            color=(:orange, 0.09))
        boxplot!(categorical_samples, lnormal_samples;
            color=(:steelblue, 1),
            show_outliers=false,
            width=0.4)
        limits!(0.5, 5.5, -2, 40)

        Axis(fig[2,2];)
        density!(lnormal_samples; 
            direction = :y,
            npoints = 1000,
            color=(:steelblue3, 0.8))
        ylims!(-2, 40)
        
        rowgap!(fig.layout, 1, 0)
        colgap!(fig.layout, 1, 0)

        rowsize!(fig.layout, 2, Relative(0.7))
        colsize!(fig.layout, 1, Relative(0.7))

        display(fig)
    end

    StatsBase.cor(unif_1, unif_2)
end


