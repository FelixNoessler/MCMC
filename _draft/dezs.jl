function DEzs_MCMC(; external_chains = 4, draws = 2000,
                   d = 4, N = 3,
                   γ = 2.38 / sqrt(2), b = 1e-4,
                   K = 10, M₀ = 10*d,
                   psnooker = 0.1)

    prior_dists = [Normal(0, 10), Normal(0, 10),
    InverseGamma(2, 3), InverseGamma(2, 3)]

    @assert M₀ > max(d, N)

    Y = DimArray(zeros(external_chains, draws, N, d),
    (chain = 1:external_chains,
        draw = 1:draws,
        internal_chain = 1:N,
        parameter = [:μ₁, :μ₂, :σ₁, :σ₂]);)

    Z = DimArray(zeros(M₀+draws*N, d),
    (draw = 1:M₀+draws*N, parameter = [:μ₁, :μ₂, :σ₁, :σ₂]);)
    X = zeros(N, d)
    xₚ = zeros(d)

    for ext_n in 1:external_chains

        for i in 1:M₀
            z = @view Z[i, :]
            prior_sample!(z, prior_dists)
        end

        X .= Z[1:N, :]
        M = M₀

        for draw in 1:draws
            for _ in 1:K
                for i in 1:N
                    r_extra = 0.0

                    if rand() < psnooker
                        r1, r2, r3 = sample(setdiff(1:M, i), 3, replace=false)

                        ######### playground
                        # we want to project Z[r1, :] and Z[r2, :] onto the line between X[i, :] and Z[r3, :]
                        proj_adj = Z[r3, :]
                        line_proj = X[i, :] - proj_adj .+ 1e-100
                        z1_proj = Z[r1, :] - proj_adj
                        z2_proj = Z[r2, :] - proj_adj

                        z1 = ((z1_proj ⋅ line_proj) / (line_proj ⋅ line_proj)) * line_proj + proj_adj
                        z2 = ((z2_proj ⋅ line_proj) / (line_proj ⋅ line_proj)) * line_proj + proj_adj

                        γₛ = rand(Uniform(1.2, 2.2))
                        proposal = vec(X[i, :] + γₛ * (z1 - z2))

                        if any(isnan.(proposal))
                            @warn "Proposal is NaN"
                            @show X[i, :]
                            @show Z[r3, :]
                            @show Z[r1, :]
                            @show Z[r2, :]
                            @show z1_proj
                            @show z2_proj
                            @show line_proj
                            @show z1
                            @show z2
                        end

                        ##########
                        z = Z[r3, :]
                        x_z = X[i, :] - z
                        D2 = max(sum(x_z .* x_z), 1.0e-300)
                        projdiff = sum((Z[r1,] -Z[r2,]) * x_z)/D2
                        @. xₚ = X[i,] + γₛ * projdiff * x_z
                        x_z = xₚ .- z
                        D2prop = max(sum(x_z .* x_z), 1.0e-300)
                        Npar12 = (d - 1)/2
                        r_extra = Npar12 * (log(D2prop) - log(D2))
                    else
                        # -------- sample r1, r2 from 1:M without i
                        r1 = rand(1:M)
                        r2 = rand(1:M)

                        while true
                            if i != r1 && i != r2 && r1 != r2
                                break
                            end
                            r1 = rand(1:M)
                            r2 = rand(1:M)
                        end

                        # -------- proposal
                        for j in 1:d
                            e = rand(Normal(0, b))
                            xₚ[j] = X[i, j] + γ * (Z[r1, j] - Z[r2, j]) + e
                        end

                    end

                    prop = fitness(xₚ, empirical_data, prior_dists)
                    old = fitness(X[i, :], empirical_data, prior_dists)
                    r = prop - old + r_extra

                    # -------- accept or reject
                    if log(rand()) < r
                        X[i, :] .= xₚ
                    end

                end # internal chains
            end # K

            Z[draw = M+1 .. M+N] = X
            M += N
            Y[chain = ext_n, draw = draw] .= X

        end # draws
    end # external_chains

    return Y
end

m = DEzs_MCMC(; psnooker = 1.0, draws = 10);

let
    draw_selected = 1:size(m, :draw) #÷ 2:size(m, :draw)
    samples = m[draw = draw_selected]

    names = ["μ₁", "μ₂", "σ₁", "σ₂"]
    fig = Figure(; resolution = (800, 900))

    for p in axes(samples, :parameter)

        Axis(fig[p,1]; title = names[p])
        density!(vec(samples[parameter = p]); color = (:blue, 0.5))

        Axis(fig[p,2])

        for ext_n in axes(samples, :chain)
            for i in axes(samples, :internal_chain)
                selected_samples = vec(samples[chain = ext_n, internal_chain = i, parameter = p])

                lines!(draw_selected, vec(selected_samples);
                    colormap = :viridis,
                    color = ext_n, colorrange = (1, size(samples, 1)))
            end
        end
    end

    fig
end



############## Snooker
### vector goes through origin
let
    v = [2, 3]
    s = [3, 2]

    hu = ((v ⋅ s) / (s ⋅ s)) * s
    @show hu = (s * s') / (s' * s) * v

    fig, ax = scatter(hu[1], hu[2]; axis=(;limits = (0, 4, 0, 4)),
                      markersize=30)
    scatter!(v[1], v[2]; color=:red, markersize=30)
    scatter!(s[1], s[2]; color=:orange, markersize=30)

    m = s[2] / s[1]
    x = 0:0.1:4
    lines!(x, m*x; color = :black)

    display(fig)
    nothing
end

### vector does not go through origin with two dimensions
let
    x1 = [2.3, 1]
    x2 = [1,2.234]
    v = [2, 1]

    m = (x2[2] - x1[2]) / (x2[1] - x1[1])
    b = x1[2] - m * x1[1]

    v_new = v - [0, b]
    s = [1, m]
    @show hu = ((v_new ⋅ s) / (s ⋅ s)) * s + [0, b]

    fig, ax = scatter(v[1], v[2]; axis=(;limits = (0, 4, 0, 4)),
                      markersize=30)
    scatter!(x1[1], x1[2]; color=:red, markersize=30)
    scatter!(hu[1], hu[2]; color=:green, markersize=30)
    x = 0:0.1:4
    lines!(x, m*x .+ b; color = :black)
    display(fig)


    x1 = [2.3, 1]
    x2 = [1, 2.234]
    v = [2, 1]

    x2_proj = x2 - x1
    v_proj = v - x1
    ((v_proj ⋅ x2_proj) / (x2_proj ⋅ x2_proj)) * x2_proj + x1

end


let
    y = [1, 3, -3]
    X = [1 0; 0 -6; 2 2];
    Q, R = qr(X)
    Q = Matrix(Q)
    Q * Q' * y
end

function normalized_orthogonal_projection(b, Z)
    # project onto the orthogonal complement of the col span of Z
    orthogonal = I - Z * inv(Z'Z) * Z'
    projection = orthogonal * b
    # normalize
    return projection / norm(projection)
end

#### three dimensions
let     # x y z
    X = [2 2 5
         1 4 3]
    z = X[:, 3]
    Xmat = hcat(ones(length(z)), X[:, 1:2])
    β = inv(Xmat'*Xmat)*Xmat'*z

    p = [1, 2, 4]

    fig, ax = scatter(X[:, 1], X[:, 2], X[:, 3]; color=:red, markersize=30)
    lines!(X[:, 1], X[:, 2], X[:, 3])
    scatter!(p[1], p[2], p[3]; color=:green, markersize=30)
    display(fig)


    p_new = p - [0, β[1], β[2]]
    # s = [1, m]
    # hu = ((v_new ⋅ s) / (s ⋅ s)) * s + [0, b]


    β
end
