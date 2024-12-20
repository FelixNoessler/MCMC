using AdaptiveParticleMCMC
using CairoMakie
using Distributions
using LabelledArrays
using Statistics
using UnPack

import MCMCChains
import PairPlots
import StatsPlots
import Random

set_theme!(
    fontsize = 18,
    Axis = (; xgridvisible = false, ygridvisible = false,
            topspinevisible = false, rightspinevisible = false),
    Legend = (; framevisible = false, titlehalign = :left,  gridshalign = :left))


function generate_data(n_observations; σ_p, σ_o, r, K, x₀)
    ts = 1:n_observations
    T = length(ts)

    s = Array{Float64}(undef, T)
    x = Array{Float64}(undef, T)
    y = Array{Float64}(undef, T)
    ε = rand(Normal(0, σ_p), T)

    for t in ts
        x_lastt = t == 1 ? x₀ : x[t-1]
        s_lastt = t == 1 ? x₀ : s[t-1]

        s[t] = (1 + r*(1 - s_lastt/K)) * s_lastt
        x[t] = (1 + r*(1 - x_lastt/K) + ε[t]) * x_lastt
        y[t] = rand(Gamma(x[t]^2 / σ_o^2, σ_o^2 / x[t]))
    end

    (; ts, s, x, y, parameter = (; σ_p, σ_o, r, K, x₀))
end

Random.seed!(123)
true_solution = generate_data(100; σ_p = 0.05, σ_o = 20.0, r = 0.1, K = 400, x₀ = 20.0);

let
    fig = Figure(size = (750, 300))

    ax = Axis(fig[1, 1]; xlabel = "time")
    scatter!(true_solution.ts, true_solution.y, color = :steelblue4, label = "observations: y")
    lines!(true_solution.ts, true_solution.x, color = :blue, label = "true hidden state: x")
    lines!(true_solution.ts, true_solution.s, color = :red, label = "process-model state: s")
    Legend(fig[1,2], ax)
    fig
end

mutable struct Particle
    s::Float64
    Particle() = new(0.0)
end


mutable struct Param
    r::Float64
    K::Float64
    σ_p::Float64
    σ_o::Float64
    x₀::Float64
end

struct ModelScratch
    par::Param
    y::Vector{Float64}
    ModelScratch() = new(Param(zeros(5)...), true_solution.y)
end


function transition!(x, rng, k, x_prev, scratch)
    @unpack r, K, σ_p, x₀ = scratch.par

    ε_t = rand(rng, Normal(0, σ_p))
    if k == 1
        x.s = (1 + r*(1 - x₀/K) + ε_t) * x₀
    else
        x.s = (1 + r*(1 - x_prev.s/K) + ε_t) * x_prev.s
    end
end


function log_potential(k, x, scratch)
    if x.s <= 0
        return -Inf
    end

    α = x.s^2 / scratch.par.σ_o^2
    θ = scratch.par.σ_o^2 / x.s
    logpdf(Gamma(α, θ), scratch.y[k])
end

function set_param!(scratch, θ)
    scratch.par.r = exp(θ.log_r)
    scratch.par.K = exp(θ.log_K)
    scratch.par.σ_p = exp(θ.log_sigma_p)
    scratch.par.σ_o = exp(θ.log_sigma_o)
    scratch.par.x₀ = exp(θ.log_x₀)
end

function prior(theta)
    (logpdf(Normal(log(0.1), 1.0), theta.log_r) +
     logpdf(Normal(log(200.0), 1.0), theta.log_K) +
     logpdf(Normal(log(0.1), 0.5), theta.log_sigma_p) +
     logpdf(Normal(log(10), 1.0), theta.log_sigma_o) +
     logpdf(Normal(log(10.0), 0.5), theta.log_x₀))
end

function sample_prior()
    LVector(log_r = rand(Normal(log(0.1), 1.0)),
            log_K = rand(Normal(log(200.0), 1.0)),
            log_sigma_p = rand(Normal(log(0.1), 0.5)),
            log_sigma_o = rand(Normal(log(10), 1.0)),
            log_x₀ = rand(Normal(log(10.0), 0.5)))
end

function sample_prior_contrain(n)
    (; r = exp.(rand(Normal(log(0.1), 1.0), n)),
       K = exp.(rand(Normal(log(200.0), 1.0), n)),
       σ_p = exp.(rand(Normal(log(0.1), 0.5), n)),
       σ_o = exp.(rand(Normal(log(10), 1.0), n)),
       x₀ = exp.(rand(Normal(log(10.0), 0.5), n)))
end


function post_pred(chn, ts; process_noise = false)
    p = get(chn; section=:parameters)
    nsamples = length(p[1])

    T = length(ts)
    X = Array{Float64}(undef, T, nsamples)

    for i in 1:nsamples
        x = Array{Float64}(undef, T)
        r = p.r[i]
        K = p.K[i]
        σ_p = p.σ_p[i]
        σ_o = p.σ_o[i]
        x₀ = p.x₀[i]

        ε = zeros(T)
        if process_noise
            ε = rand(Normal(0, σ_p), T)
        end

        for t in ts
            x_lastt = t == 1 ? x₀ : x[t-1]
            x[t] = (1 + r*(1 - x_lastt/K) + ε[t]) * x_lastt
        end
        X[:, i] = x
    end

    mapslices(x_t -> quantile(x_t, [0.025, 0.25, 0.5, 0.75, 0.975]), X, dims = 2)
end

function sample_hiddenstate(data, n)
    T, nsamples, nchains = size(data)
    ntotalsamples = nsamples * nchains
    d = deepcopy(data)
    d = reshape(d, (T, ntotalsamples))
    d[:, sample(1:ntotalsamples, n; replace = false)]
end


##################################################
nsamples = 100_000

post_pmcmc, hidden_state = let
    T = length(true_solution.y)
    nparticles = 100
    nchains = 4

    post_objs = []
    hidden_states_obj = []
    for i in 1:nchains
        theta0 = sample_prior()
        state = SMCState(T, nparticles, Particle, ModelScratch, set_param!,
                         log_potential, transition!);
        out = adaptive_pmmh(theta0, prior, state, nsamples; thin = 1,
                            save_paths = true, b = 0, show_progress = true);

        S = [out.X[j][i].s for i = 1:length(out.X[1]), j = 1:length(out.X)]
        push!(hidden_states_obj, S)

        θ = deepcopy(out.Theta)
        θ[1, :] = exp.(out.Theta[1, :])
        θ[2, :] = exp.(out.Theta[2, :])
        θ[3, :] = exp.(out.Theta[3, :])
        θ[4, :] = exp.(out.Theta[4, :])
        θ[5, :] = exp.(out.Theta[5, :])

        push!(post_objs, θ')
    end

    cat(post_objs..., dims = 3), cat(hidden_states_obj..., dims = 3)
end;


burnin = nsamples ÷ 3
thin = 100
chn_pmcmc = MCMCChains.Chains(post_pmcmc[burnin:thin:end, :, :], collect(fieldnames(Param)))

display(StatsPlots.plot(chn_pmcmc))


PairPlots.pairplot(chn_pmcmc, PairPlots.Truth( true_solution.parameter))

PairPlots.pairplot(chn_pmcmc[:, :, 1], chn_pmcmc[:, :, 2],
                   chn_pmcmc[:, :, 3], chn_pmcmc[:, :, 4])

let
    n = 1000
    prior_df = (;
        r = exp.(rand(Normal(log(0.1), 1.0), n)),
        K = exp.(rand(Normal(log(200.0), 1.0), n)),
        σ_p = exp.(rand(Normal(log(0.1), 0.5), n)),
        σ_o = exp.(rand(Normal(log(10), 1.0), n)),
        x₀ = exp.(rand(Normal(log(10.0), 0.5), n)))

    fig = PairPlots.pairplot(
        PairPlots.Series(prior_df, label = "prior", color = (:black, 0.4)),
        PairPlots.Series(chn_pmcmc, label = "posterior", color = (:red, 0.5)))

    display(fig)
end




q95 = mapslices(x -> quantile(x, [0.025, 0.975]), hidden_state, dims=(2,3))
q5 = mapslices(x -> quantile(x, [0.25, 0.75]), hidden_state, dims=(2,3))
q_median = mapslices(median, hidden_state, dims=(2,3))

q_post = post_pred(chn_pmcmc, true_solution.ts; process_noise = false)
q_post1 = post_pred(chn_pmcmc, true_solution.ts; process_noise = true)
hidden_state_samples = sample_hiddenstate(hidden_state, 50)

let
    fig = Figure(size = (1200, 1500))

    pax1 = Axis(fig[1, 1]; ylabel = "particle filter", xticklabelsvisible = false)
    for i in 1:size(hidden_state_samples)[2]
        scatter!(true_solution.ts, hidden_state_samples[:, i], color = (:black, 0.5),
                 markersize = 3)
    end
    lines!(true_solution.ts, true_solution.x, color = :blue,
           label = "true hidden state: x")

    pax2 = Axis(fig[1, 2]; yticklabelsvisible = false, xticklabelsvisible = false)
    b1 = band!(true_solution.ts, q95[:, 1], q95[:, 2], color = (:black, 0.2),
                label = "95% credible interval")
    b2 = band!(true_solution.ts, q5[:, 1], q5[:, 2], color = (:black, 0.5),
                label = "50% credible interval")
    m = lines!(true_solution.ts, q_median[:, 1], color = :black,
                label = "median")
    lines!(true_solution.ts, true_solution.x, color = :blue,
            label = "true hidden state: x")
    linkyaxes!(pax1, pax2)

    max1 = Axis(fig[2, 1]; ylabel = "model", xticklabelsvisible = false)
    for i in 1:50
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(chn_pmcmc, 1).value
        x = p[var = :x₀][1]
        r = p[var = :r][1]
        K = p[var = :K][1]

        for t in true_solution.ts
            x = (1 + r*(1-x/K)) * x
            xs[t] = x
        end

        global draw = lines!(true_solution.ts, xs, color = (:black, 0.2))
    end
    mod = lines!(true_solution.ts, true_solution.s, color = :red, linewidth = 3)

    max2 = Axis(fig[2, 2]; yticklabelsvisible = false, xticklabelsvisible = false)
    band!(true_solution.ts, q_post[:, 1], q_post[:, 5], color = (:black, 0.2),
          label = "95% credible interval")
    band!(true_solution.ts, q_post[:, 2], q_post[:, 4], color = (:black, 0.5),
          label = "50% credible interval")
    lines!(true_solution.ts, q_post[:, 3], color = :black, label = "median")
    mod = lines!(true_solution.ts, true_solution.s, color = :red, linewidth = 3)
    linkyaxes!(max1, max2)


    Axis(fig[3, 1]; ylabel = "state", xticklabelsvisible = false)
    for i in 1:50
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(chn_pmcmc, 1).value
        x = p[var = :x₀][1]
        r = p[var = :r][1]
        K = p[var = :K][1]
        σ_p = p[var = :σ_p][1]


        ε_t_dist = Normal(0, σ_p)
        ε = rand(ε_t_dist, length(true_solution.ts))

        for t in true_solution.ts
            x = (1 + r*(1-x/K) + ε[t]) * x
            xs[t] = x
        end

        draw_err = lines!(true_solution.ts, xs, color = (:black, 0.2))
    end
    st = lines!(true_solution.ts, true_solution.x, color = :blue, linewidth = 3)

    Axis(fig[3, 2]; yticklabelsvisible = false)
    band!(true_solution.ts, q_post1[:, 1], q_post1[:, 5], color = (:black, 0.2),
          label = "95% credible interval")
    band!(true_solution.ts, q_post1[:, 2], q_post1[:, 4], color = (:black, 0.5),
            label = "50% credible interval")
    lines!(true_solution.ts, q_post1[:, 3], color = :black, label = "median")
    st = lines!(true_solution.ts, true_solution.x, color = :blue, linewidth = 3)


    Axis(fig[4, 1]; xlabel = "time", ylabel = "observations")
    for i in 1:50
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(chn_pmcmc, 1).value
        x = p[var = :x₀][1]
        r = p[var = :r][1]
        K = p[var = :K][1]
        σ_p = p[var = :σ_p][1]
        σ_o = p[var = :σ_o][1]

        ε_t_dist = Normal(0, σ_p)
        ε = rand(ε_t_dist, length(true_solution.ts))

        for t in true_solution.ts
            x = (1 + r*(1-x/K) + ε[t]) * x

            if x > 0
                xs[t] = rand(Gamma(x^2 / σ_o^2, σ_o^2 / x))
            else
                xs[t] = NaN
            end
        end

        global obs_gen = scatter!(true_solution.ts, xs, color = (:black, 0.3),
                                  markersize = 3)
    end
    obs = scatter!(true_solution.ts, true_solution.y, color = :steelblue4, linewidth = 5)

    Legend(fig[4, 2],
           [[mod, st, obs],
            [b1, b2, m, draw,
             MarkerElement(marker = :circle, markersize = 8, color = (:black, 0.5))]],
           [["Model", "Hidden state", "Observations"],
            ["95% credible interval", "50% credible interval", "Median",
             "Draws", "Generated observations"]],
           ["Underlying data/model", "Particle filter and model estimation"];
           tellwidth = false)

    colgap!(fig.layout, 1, 0)
    [rowgap!(fig.layout, i, 0) for i in 1:3]
    display(fig)
end


##################################################
# prior predictive check

p_samples = sample_prior_contrain(10000)
p_chain = MCMCChains.Chains(hcat(collect(p_samples)...), collect(keys(p_samples)))

let
    fig = Figure(size = (900, 900))

    Axis(fig[1, 1]; ylabel = "model")
    for i in 1:250
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(p_chain, 1).value
        x = p[var = :x₀][1]
        r = p[var = :r][1]
        K = p[var = :K][1]

        for t in true_solution.ts
            x = (1 + r*(1-x/K)) * x
            xs[t] = x
        end

        global draw = lines!(true_solution.ts, xs, color = (:black, 0.1))
    end
    mod = lines!(true_solution.ts, true_solution.s, color = :red, linewidth = 3)


    Axis(fig[2, 1]; ylabel = "state")
    for i in 1:250
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(p_chain, 1).value
        x = p[var = :x₀][1]
        r = p[var = :r][1]
        K = p[var = :K][1]
        σ_p = p[var = :σ_p][1]


        ε_t_dist = Normal(0, σ_p)
        ε = rand(ε_t_dist, length(true_solution.ts))

        for t in true_solution.ts
            x = (1 + r*(1-x/K) + ε[t]) * x
            xs[t] = x
        end

        global draw_err = lines!(true_solution.ts, xs, color = (:black, 0.1))
    end
    st = lines!(true_solution.ts, true_solution.x, color = :blue, linewidth = 3)

    Axis(fig[3, 1]; xlabel = "time", ylabel = "observations")
    for i in 1:250
        xs = Array{Float64}(undef, length(true_solution.ts))

        p = sample(p_chain, 1).value
        x = p[var = :x₀][1]
        r = p[var = :r][1]
        K = p[var = :K][1]
        σ_p = p[var = :σ_p][1]
        σ_o = p[var = :σ_o][1]

        ε_t_dist = Normal(0, σ_p)
        ε = rand(ε_t_dist, length(true_solution.ts))

        for t in true_solution.ts
            x = (1 + r*(1-x/K) + ε[t]) * x

            if x > 0
                xs[t] = rand(Gamma(x^2 / σ_o^2, σ_o^2 / x))
            else
                xs[t] = NaN
            end
        end

        global obs_gen = scatter!(true_solution.ts, xs, color = (:black, 0.1),
                                  markersize = 3)
    end
    obs = scatter!(true_solution.ts, true_solution.y, color = :steelblue4, linewidth = 5)


    Legend(fig[1:3, 2],
           [[mod, st, obs],
            [draw, draw_err, MarkerElement(marker = :circle, markersize = 8,
                                           color = (:black, 0.5))]],
           [["Model", "Hidden state", "Observations"],
            ["Draws", "Draws with process error", "Generated observations"]],
           ["Underlying data/model", "Model estimation"])

    display(fig)
end
