using CairoMakie
using OrdinaryDiffEq
using Distributions
using Statistics, LinearAlgebra, Random

own_theme = Theme(
    fontsize=18,
    Axis=(xgridvisible=false, ygridvisible=false,
          topspinevisible=false, rightspinevisible=false),
)
set_theme!(own_theme)

function lotka_volterra(du, u, p, t)
    α, β, γ, δ = p
    x, y = u
    
    du[1] = (α - β * y) * x 
    du[2] = (δ * x - γ) * y 

    return nothing
end

function generate_data(; plot=false)
    u0 = [1.0, 1.0]
    p = [1.5, 1.0, 3.0, 1.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)
    
    sol = solve(prob, Tsit5();
                saveat=0.1)
    
    estim_t = sol.t
    estim_dat = Array(sol) .+ rand(Normal(0, 0.5), size(Array(sol)))
    
    sol = solve(prob, Tsit5();
        saveat=0.01)
    
    if plot
        fig = Figure()
        ax = Axis(fig[1,1])
        lines!(sol.t, sol[1, :];
               color=:black, linewidth=3)
        lines!(sol.t, sol[2, : ];
            color=:black, linewidth=3)
        
        scatter!(estim_t, estim_dat[1,:])
        scatter!(estim_t, estim_dat[2,:])
        
        # display(fig)
        
        return estim_t, estim_dat, fig, ax
        
    end
    
    return estim_t, estim_dat
end



function unnormalized_posterior(θ, prior_dists, data)
    σ, α, β, γ, δ = θ
    nparameter = length(θ)
    
    ## prior 
    if σ <= 0
        return -Inf
    end
    
    prior = 0
    for i in 1:nparameter
        prior += logpdf(prior_dists[i], θ[i])
    end
    if prior == -Inf
        return -Inf
    end

    ## likelihood
    p = [α, β, γ, δ]
    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)
    predicted = solve(prob, Tsit5(); p=p, saveat=0.1)
    
    likelihood = 0
    for i in 1:length(predicted)
        likelihood += logpdf(MvNormal(predicted[i], σ^2 * I), data[:, i])
    end
    
    return prior + likelihood
end


let 
    rng = MersenneTwister(0)
    estim_t, estim_dat, data_fig, data_axes = generate_data(; plot=true)
    
    ##### priors
    σ_prior = InverseGamma(2, 3)
    α_prior = truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β_prior = truncated(Normal(1.2, 0.5); lower=0, upper=2)
    γ_prior = truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ_prior = truncated(Normal(1.0, 0.5); lower=0, upper=2)
    prior_dists = [σ_prior, α_prior, β_prior, γ_prior, δ_prior]
    
    nsamples = 80_000
    nchains = 5
    nparameter = 5
    accepted_θ = zeros(nchains, nparameter, nsamples)
    accepted = zeros(nchains)
    θ = zeros(nchains, nparameter)
    
    @Threads.threads for n in 1:nchains        
        ## start values for the parameters in the chain
        # for i in 1:nparameter
        #     θ[n, i] = rand(rng, prior_dists[i])
        # end
        θ[n, :] = [0.5, 1.6, 1.1, 2.8, 0.95] + rand(rng, Normal(0, 0.1), 5)
        post = unnormalized_posterior(θ[n, :], prior_dists, estim_dat)
        
        for k in 1:nsamples
            ## new proposal
            proposal_dist = MvNormal(θ[n, :], Diagonal([0.1, 0.2, 0.1, 0.2, 0.1] ./ 200))
            θstar = rand(rng, proposal_dist) 
            
            ## evaluate prior + likelihood
            poststar = unnormalized_posterior(θstar, prior_dists, estim_dat)
            
            ## Metropolis ratio
            ratio = poststar - post

            if log(rand()) < min(ratio, 1)
                accepted[n] += 1
                θ[n, :] = θstar
                post = poststar
            end
            
            accepted_θ[n, :, k] = θ[n, :]
        end
 
    end
    
    burnin = max(Int(0.5*nsamples), 1)
    @show median_σ = median(accepted_θ[:, 1, burnin:end])
    @show median_α = median(accepted_θ[:, 2, burnin:end])
    @show median_β = median(accepted_θ[:, 3, burnin:end])
    @show median_γ = median(accepted_θ[:, 4, burnin:end])
    @show median_δ = median(accepted_θ[:, 5, burnin:end])
    @show accepted / nsamples

    #### trace plots and densities
    begin
        fig = Figure(resolution=(800, 800))
        
        titles = ["σ", "α", "β", "γ", "δ"]
        for i in 1:nparameter
            Axis(fig[i,1]; title = titles[i])
            
            for n in 1:nchains
                lines!((burnin:nsamples) .- burnin, accepted_θ[n, i, burnin:end];
                    color=(Makie.wong_colors()[n], 0.5))
            end
            
            Axis(fig[i,2])
            for n in 1:nchains
                density!(accepted_θ[n, i, burnin:end];
                        bins=20, 
                        color= (Makie.wong_colors()[n], 0.05),
                        strokecolor = (Makie.wong_colors()[n], 1),
                        strokewidth = 2, strokearound = false)
            end
       
        end
        rowgap!(fig.layout, 1, 5)
        
        display(fig)
        # save("img/trace_unknown_sigma_mu.png", fig);
    end
    
    ### posterior predictive check
    npost_samples = 200
    
    selected_chains = rand(1:nchains, npost_samples)
    selected_samples = rand(burnin:nsamples, npost_samples)
    
    
    fig, ax = data_fig, data_axes
    
    for k in 1:npost_samples
        θ = accepted_θ[selected_chains[k], :, selected_samples[k]]
        p = θ[2:5]
        u0 = [1.0, 1.0]
        tspan = (0.0, 10.0)
        prob = ODEProblem(lotka_volterra, u0, tspan, p)
        sol = solve(prob, Tsit5(); p=p, saveat=0.01)
        
        lines!(ax, sol.t, sol[1, :], color=(Makie.wong_colors()[1], 0.1))
        lines!(ax, sol.t, sol[2, :], color=(Makie.wong_colors()[2], 0.1))
    end
    
    display(fig)
    
end