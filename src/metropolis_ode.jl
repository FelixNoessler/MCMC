# # Metropolis-Algorithm for parameters estimation of ODEs
#
# the script is adapted from the [Bayesian Estimation of Differential Equations](https://turing.ml/stable/tutorials/10-bayesian-differential-equations/) tutorial from Turing.jl, but instead of relying on the Nuts algorithm of Turing.jl, a simple Metroplis algorithm is coded here from scratch

# Load packages and Makie theme
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


# Define ODE-System
function lotka_volterra(du, u, p, t)
    α, β, γ, δ = p
    x, y = u
    
    du[1] = (α - β * y) * x 
    du[2] = (δ * x - γ) * y 

    return nothing
end

# Generate a test data set
function generate_data(rng; p)
    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)
    
    sol = solve(prob, Tsit5();
                saveat=0.1)
    estim_dat = Array(sol) .+ rand(rng, Normal(0, 0.5), size(Array(sol)))
    
    return estim_dat, sol.t
end

# Function to calculate the unnormalized posterior density
function unnormalized_posterior(θ, prior_dists, data, t)
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
    predicted = solve(prob, Tsit5(); p=p, saveat=t)

    likelihood = 0
    for i in 1:length(predicted)
        likelihood += logpdf(MvNormal(predicted[i], σ^2 * I), data[:, i])
    end
    
    return prior + likelihood
end

# Function to simulate the Markov chains
function run_chains(rng, data, t; 
                    σ_prop,
                    nchains=5,
                    nsamples=5_000)

    ## priors
    σ_prior = truncated(InverseGamma(2, 3); lower=0, upper=1)
    α_prior = truncated(Normal(1.5, 0.5); lower=0.8, upper=2.5)
    β_prior = truncated(Normal(1.2, 0.5); lower=0, upper=2)
    γ_prior = truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ_prior = truncated(Normal(1.0, 0.5); lower=0, upper=2)
    prior_dists = [σ_prior, α_prior, β_prior, γ_prior, δ_prior]
    
    nparameter = 5
    accepted_θ = zeros(nchains, nparameter, nsamples)
    accepted = zeros(nchains)
    θ = zeros(nchains, nparameter)
    
    Threads.@threads for n in 1:nchains        
        ## start values for the parameters in the chain
        ## rough guesses are used here 
        ## it would also possible to use the prior distributions as follows: 
        ## for i in 1:nparameter
        ##     θ[n, i] = rand(rng, prior_dists[i])
        ## end
        θ[n, :] = [0.7, 1.4, 0.9, 3.1, 1.1] .+ rand(rng, Normal(0, 0.1), 5)
        post = unnormalized_posterior(θ[n, :], prior_dists, data, t)
        
        for k in 1:nsamples
            ## new proposal
            proposal_dist = MvNormal(θ[n, :], σ_prop)
            θstar = rand(rng, proposal_dist) 
            
            ## evaluate prior + likelihood
            poststar = unnormalized_posterior(θstar, prior_dists, data, t)
            
            ## Metropolis ratio
            ratio = poststar - post

            if log(rand(rng)) < min(ratio, 1)
                accepted[n] += 1
                θ[n, :] = θstar
                post = poststar
            end
            
            accepted_θ[n, :, k] = θ[n, :]
        end
 
    end
    
    return accepted_θ, accepted / nsamples
end

# trace plots and densities
function plot_trace_dens(; θ, burnin=nothing)
    fig = Figure(resolution=(800, 800))
    
    titles = ["σ", "α", "β", "γ", "δ"]
    nchains, nparameter, nsamples = size(θ)
    burnin = isnothing(burnin) ? max(Int(0.5*nsamples), 1) : burnin
    
    for i in 1:nparameter
        Axis(fig[i,1]; title = titles[i])
        
        for n in 1:nchains
            lines!((burnin:nsamples) .- burnin, θ[n, i, burnin:end];
                color=(Makie.wong_colors()[n], 0.5))
        end
        
        Axis(fig[i,2])
        for n in 1:nchains
            density!(θ[n, i, burnin:end];
                    bins=20, 
                    color= (Makie.wong_colors()[n], 0.05),
                    strokecolor = (Makie.wong_colors()[n], 1),
                    strokewidth = 2, strokearound = false)
        end

    end
    rowgap!(fig.layout, 1, 5)
    
    display(fig)
    save("img/ode_trace.png", fig)

    return nothing
end

# posterior predictive check
function posterior_check(rng; θ, data, t, p, npost_samples=500, burnin=nothing)

    nchains, nparameter, nsamples = size(θ)
    burnin = isnothing(burnin) ? max(Int(0.5*nsamples), 1) : burnin
    
    u0 = [1.0, 1.0]
    tspan = (0.0, 10.0)
    
    fig = Figure()
    Axis(fig[1,1];
         xlabel="Time",
         ylabel="Density")
    
    ## select posterior draws and plot solutions
    selected_chains = rand(rng, 1:nchains, npost_samples)
    selected_samples = rand(rng, burnin:nsamples, npost_samples)
    for k in 1:npost_samples
        θi = θ[selected_chains[k], :, selected_samples[k]]
        p_i = θi[2:5]
 
        prob = ODEProblem(lotka_volterra, u0, tspan, p_i)
        sol = solve(prob, Tsit5(); saveat=0.01)
        
        lines!(sol.t, sol[1, :], color=(Makie.wong_colors()[1], 0.05))
        lines!(sol.t, sol[2, :], color=(Makie.wong_colors()[2], 0.05))
    end
    
    ## true solution
    prob = ODEProblem(lotka_volterra, u0, tspan, p)
    sol = solve(prob, Tsit5(); p=p, saveat=0.01)
    
    lines!(sol.t, sol[1, :], 
           color=:black,
           linewidth=2)
    lines!(sol.t, sol[2, :], 
           color=:black,
           linewidth=2)
    
    ## measured data
    scatter!(t, data[1, :])
    scatter!(t, data[2, :])

    display(fig)
    save("img/ode_pred.png", fig)
    
    return nothing
end



# Run everything
let 
    rng = MersenneTwister(123)
    
    ## "true" parameter values
    p = [1.5, 1.0, 3.0, 1.0]
    data, t = generate_data(rng; p)
    
    ## Simulate.
    σ_prop = Diagonal([0.001, 0.001, 0.001, 0.001, 0.001])
    θ, acceptance_rate = run_chains(rng, data, t;
                                    σ_prop,
                                    nsamples=200_000)
    ## Plot.
    plot_trace_dens(; θ, burnin=50_000)
    posterior_check(rng; θ, data, t, p, burnin=50_000)
end

# **Results:**
#
# ![](../img/ode_trace.png)
#
# the black line is generated with the parameters that are also used to produce the test data set, orange and blue lines are produced with posterior draws, the circles represent the test data set:
# ![](../img/ode_pred.png)

## to generate the Markdown file from the script:
import Literate; Literate.markdown("src/metropolis_ode.jl", "md/"; flavor=Literate.CommonMarkFlavor(), execute=false)