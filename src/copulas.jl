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


function generate_Ω(v::Vector{Vector{Float64}})
    n = size(v)[1]+1
    m = [ j>i ? v[i][j-i] : 0 for i=1:n, j=1:n ]
    m = Diagonal(ones(n)) .+ m
    return Hermitian(m, :U)
end

function generate_Σ(Ω::Matrix, σs=ones(size(Ω)[1]))
    Σ = Diagonal(σs) * Ω * Diagonal(σs)
    return Σ
end 

function generate_Σ(ρs::Vector{Vector{Float64}}, σs=ones(size(ρs)[1]+1))
    Ω = generate_Ω(ρs)
    Σ = Diagonal(σs) * Ω * Diagonal(σs)
    return Σ
end 

function Σ_proposal(Σ; σ=0.05)
    proposal(μ) = rand(truncated(Normal(μ, σ), lower=-1, upper=1))

    n = size(Σ)[1]
    m = [ j>i ? proposal(Σ[i, j]) : Σ[i, j] for i=1:n, j=1:n ]
    
    return Hermitian(m)
end


function positive_proposal(θ; σ=0.1)
    return rand(truncated(Normal(θ, σ); lower=0))
end



function gaussian_copula_gendata(
    nsamples, 
    ρs_Ω, 
    margin_dists...)

    n_ρ = length(ρs)+1
    n = length(margin_dists)

    if n_ρ != n
        throw(ArgumentError("Arguments don't have the right dimensions: $n marginal distributions and $(n_ρ-1) vectors of ρ for the correlation matrix"))
    end

    Σ = generate_Σ(ρs_Ω)
    D = MultivariateNormal(zeros(n), Hermitian(Σ))
    normal_samples = rand(D, nsamples)
    copula_samples = similar(normal_samples)

    for i in 1:n
        unif_samples = cdf.(Normal(0, 1), normal_samples[i, :])
        copula_samples[i, :] = quantile(margin_dists[i], unif_samples)
    end

    return copula_samples
end


function unnormaliized_posterior(; θ, data, prior_dists)
    margin_dists = [Beta(θ[:α], θ[:β]), Gamma(θ[:k], θ[:β_gamma])]
    ndim = size(data)[1] 

    ### p(θ) × p(data|θ) unnormalized posterior
    p = 0

    ### p(θ) prior probability
    for i in 1:5
        p += logpdf(prior_dists[i], θ[i])
    end

    ### p(data|θ) likelihood probability of marginal dists
    for i in 1:ndim
        p += sum( logpdf.(margin_dists[i], data[i, :]) )
    end

    #### p(data|θ) likelihood probability of Σ
    # backtransformation of the data
    # margin_dists = [Beta(5, 7), Gamma(2,3)]

    data_joint_normal = similar(data)
    for i in 1:ndim
        data_cdf = cdf.(margin_dists[i], data[i, :])
        data_joint_normal[i, :] = quantile(Normal(0, 1), data_cdf)
    end
    p += sum( logpdf(MvNormal(zeros(ndim), θ[:Σ]), data_joint_normal) )

    return p
end

function copula_inference(data; nsamples = 1_000, nchains = 1, remove_burnin=true)
    ndim = size(data)[1] 

    ###### prior
    prior_dists = [ 
        LKJ(ndim, 2.0), 
        Gamma(2,3), Gamma(2,3),
        Gamma(2,3), Gamma(2,3)
    ]

    accepted_θ = (
            Σ=fill([NaN NaN; NaN NaN], nchains, nsamples),
            α=fill(NaN, nchains, nsamples), 
            β=fill(NaN, nchains, nsamples), 
            k=fill(NaN, nchains, nsamples), 
            β_gamma=fill(NaN, nchains, nsamples) 
    ) 

    accepted = zeros(nchains)


    Threads.@threads for n in 1:nchains

        θ = (
            Σ=rand(prior_dists[1]), 
            α=rand(prior_dists[2]), β=rand(prior_dists[3]),
            k=rand(prior_dists[4]), β_gamma=rand(prior_dists[5]))

        post = unnormaliized_posterior(; θ, data, prior_dists)

        for k in 1:nsamples
            θstar = (
                Σ=Σ_proposal(θ[:Σ]), 
                α=positive_proposal(θ[:α]), β=positive_proposal(θ[:β]),
                k=positive_proposal(θ[:k]), β_gamma=positive_proposal(θ[:β_gamma])
            )

            ## evaluate prior + likelihood
            poststar = unnormaliized_posterior(; θ=θstar, data, prior_dists)

            ## M-H ratio
            ratio = poststar - post

            if log(rand()) < min(ratio, 1)
                accepted[n] += 1
                θ = θstar
                post = poststar
            end

            ## write the parameter values into the final named tuple
            for p_name in keys(θ)
                accepted_θ[p_name][n,k] = θ[p_name]
            end
        end
    end

    burnin = Int(2/4 *nsamples)
    @show accepted / nsamples

    if remove_burnin
        vals = []
        for (i, p_name) in enumerate(keys(accepted_θ))
            push!(vals, accepted_θ[p_name][:, burnin+1:end])
        end

        return (; zip(keys(accepted_θ), vals, )...  )
    end

    return accepted_θ 
end

sim_data = gaussian_copula_gendata(500, [[0.0]], Beta(5, 7), Gamma(2,3))
chains = copula_inference(sim_data; nsamples=20000, nchains=4);

getindex.(chains[:Σ], 2)
median(chains[:α])
median(chains[:β])
median(chains[:k])
median(chains[:β_gamma])




huhu


# let 
#     extrema_1 = extrema(sim_data[1, :]) .+ [-3, 3]
#     extrema_2 = [0, 1]

#     fig = Figure(resolution=(1000,800))
#     Axis(fig[1,1])
#     density!(sim_data[1, :];
#         color=(:steelblue3, 0.8))
#     xlims!(extrema_1...)

#     Axis(fig[2,1])
#     scatter!(sim_data[1, :], sim_data[2, :];
#         color=(:orange, 0.5))
#     limits!(extrema_1..., extrema_2...)

#     Axis(fig[2,2];)
#     density!(sim_data[2, :]; 
#         direction = :y,
#         npoints = 1000,
#         color=(:steelblue3, 0.8))
#     ylims!(extrema_2...)
    
#     rowgap!(fig.layout, 1, 0)
#     colgap!(fig.layout, 1, 0)

#     rowsize!(fig.layout, 2, Relative(0.7))
#     colsize!(fig.layout, 1, Relative(0.7))

#     display(fig)
# end





#     ############# Gaussian copula
#     ρ = -0.99
#     Σ = [1 ρ; ρ 1] # σ1, σ2 = 1, 1 
#     d = MultivariateNormal([0, 0], Hermitian(Σ))

#     normal_samples = rand(d, 20000)
    
#     norm_margin1 = Normal(0, 1)
#     norm_margin2 = Normal(0, 1)

#     unif_1 = cdf.(norm_margin1, normal_samples[1, :])
#     unif_2 = cdf.(norm_margin2, normal_samples[2, :])

#     ############# Archimedean copula: Clayton
#     θ = 8
#     # generate a sample
#     u = rand(10000)
#     t = rand(10000)
#     v = @. ((t / u^(-θ-1))^(-θ/(1+θ)) - u^(-θ) + 1)^(-1/θ)

#     scatter(u, v;
#         color=(:steelblue, 0.2))
#     unif_1 = u
#     unif_2 = v
    
#     begin
#         fig = Figure()
#         Axis(fig[1,1])
#         hist!(unif_1; bins=50)
        
#         Axis(fig[2,1])
#         scatter!(unif_1, unif_2;
#             color=(:orange, 0.2))


#         Axis(fig[2,2])
#         hist!(unif_2; bins=50, 
#             direction =:x)

#         rowgap!(fig.layout, 1, 0)
#         colgap!(fig.layout, 1, 0)

#         rowsize!(fig.layout, 2, Relative(0.7))
#         colsize!(fig.layout, 1, Relative(0.7))

#         display(fig)
#     end

     

#     ############ first example: Beta and Gamma
#     gamma_dist = Gamma(2, 5)
#     gamma_samples = quantile(gamma_dist, unif_1)

#     beta_dist = Beta(5, 10)
#     beta_samples = quantile(beta_dist, unif_2)


#     begin
#         fig = Figure()
#         Axis(fig[1,1])
#         density!(gamma_samples;
#             color=(:steelblue3, 0.8))
#         xlims!(-5, 65)

#         Axis(fig[2,1])
#         scatter!(gamma_samples, beta_samples;
#             color=(:orange, 0.2))
#         limits!(-5, 65, -0.1, 1.1)

#         Axis(fig[2,2];)
#         density!(beta_samples; 
#             direction = :y,
#             npoints = 1000,
#             color=(:steelblue3, 0.8))
#         ylims!(-0.1, 1.1)
        
#         rowgap!(fig.layout, 1, 0)
#         colgap!(fig.layout, 1, 0)

#         rowsize!(fig.layout, 2, Relative(0.7))
#         colsize!(fig.layout, 1, Relative(0.7))

#         display(fig)
#     end



#     ############ first example: Categorical and LogNormal
#     categorical_dist = Categorical([0.2, 0.1, 0.1, 0.4, 0.2])
#     categorical_samples = quantile(categorical_dist, unif_1)
#     discrete_counts = StatsBase.counts(categorical_samples, 1:5)
    
#     lognormal_dist = LogNormal(2, 0.5)
#     lnormal_samples = quantile(lognormal_dist, unif_2)


#     begin
#         fig = Figure()
#         Axis(fig[1,1])
#         barplot!(1:5, discrete_counts;
#             color=(:steelblue3, 0.8))
#         xlims!(0.5, 5.5)

#         Axis(fig[2,1])
#         jitter = (rand(length(categorical_samples)) .- 0.5) .* 0.8
#         scatter!(categorical_samples .+ jitter, lnormal_samples;
#             color=(:orange, 0.09))
#         boxplot!(categorical_samples, lnormal_samples;
#             color=(:steelblue, 1),
#             show_outliers=false,
#             width=0.4)
#         limits!(0.5, 5.5, -2, 40)

#         Axis(fig[2,2];)
#         density!(lnormal_samples; 
#             direction = :y,
#             npoints = 1000,
#             color=(:steelblue3, 0.8))
#         ylims!(-2, 40)
        
#         rowgap!(fig.layout, 1, 0)
#         colgap!(fig.layout, 1, 0)

#         rowsize!(fig.layout, 2, Relative(0.7))
#         colsize!(fig.layout, 1, Relative(0.7))

#         display(fig)
#     end

#     StatsBase.cor(unif_1, unif_2)
# end

# using Copulas

# let
#     X₁ = Gamma(2,3)
#     X₂ = DiscreteNonParametric([200, 400, 600], [0.2, 0.6, 0.2])
#     X₃ = LogNormal(0,1)
#     C = ClaytonCopula(3,0.7)
#     D = SklarDist(C,(X₁,X₂,X₃)) 

#     simu = rand(D,1000)
#     D_estim = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Categorical,LogNormal}}, simu)

    
# end

# let 
#     ρ = -0.7
#     Σ = [1 ρ; ρ 1]
#     C = Copulas.GaussianCopula(Σ)
#     X₁ = Beta(7, 3)
#     X₂ = Gamma(2,3)
#     D = SklarDist(C, (X₁, X₂))

#     simu = rand(D,1000)
# end