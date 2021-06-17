using Random
using Distributions

function ϕ(h::Float64)::Float64
    return 10 * h ^ 2
end

function p(h::Float64, o::Float64)::Float64
    return 1 - ((h - o)^4/(2^4))
end

function f(τ::Float64, h::Float64, o::Float64)
    prob = p(h, o)
#     @show prob
#     @show τ
    return clamp(τ + sqrt(prob * τ), 0, 1)
    #max(min(1, τ + sqrt(prob * τ)), 0)
end

function g(τ::Float64, h::Float64, o::Float64)
    prob = p(h, o)
    return clamp(τ - sqrt((1 - prob) * (1 - τ)), 0, 1)
    #return max(min(1, τ - sqrt((1 - prob) * (1 - τ)), 0))
end


function shift_opinion(h::Float64, o::Float64, α::Float64)::Float64
    return α * h + (1 - α) * o
end

function V(T::Int64, τ::Float64, h::Float64,
    C::Array{Array{Float64, 1}, 1}, α::Float64, f, g)::Float64
    if T == 0
        return ϕ(h)
    end
    leave = (1 - τ) * U(T, τ, h)
    remain = τ * W(T, τ, h, C, α, f, g)
    return leave + remain
end

function V(T::Int64, τ::Float64, h::Float64,
    C::Array{Array{Float64, 1}, 1}, α::Float64, f, g, heuristic)::Float64
    if T == 0
        return ϕ(h)
    end
    leave = (1 - τ) * U(T, τ, h)
    remain = τ * W(T, τ, h, C, α, f, g, heuristic)
    return leave + remain
end

function U(T::Int64, τ::Float64, h::Float64)::Float64
    return T * ϕ(h)
end

function W(T::Int64, τ::Float64, h::Float64,
    C::Array{Array{Float64, 1}, 1}, α::Float64, f, g)::Float64
    curr_cost = Inf
    for oᵢ ∈ C[T]
        #oᵢ = C[T][i]
        pᵢ = p(h, oᵢ)

        τ_bad = g(τ, h, oᵢ)
        τ_good = f(τ, h, oᵢ)

        h′ = shift_opinion(h, oᵢ, α)

        positive = pᵢ * (ϕ(h′) + V(T - 1, τ_good, h′, C, α, f, g))
        negative = (1 - pᵢ) * (ϕ(h) + V(T - 1, τ_bad, h, C, α, f, g))
        total_cost = positive + negative
        if total_cost < curr_cost
            curr_cost = total_cost
        end
    end
    return curr_cost
end

function W(T::Int64, τ::Float64, h::Float64,
    C::Array{Array{Float64, 1}, 1}, α::Float64, f, g, heuristic)::Float64
    i = heuristic(τ, h, C[T], α, f, g)
    oᵢ = C[T][i]
    pᵢ = p(h, oᵢ)

    τ_bad = g(τ, h, oᵢ)
    τ_good = f(τ, h, oᵢ)

    h′ = shift_opinion(h, oᵢ, α)

    positive = pᵢ * (ϕ(h′) + V(T - 1, τ_good, h′, C, α, f, g, heuristic))
    negative = (1 - pᵢ) * (ϕ(h) + V(T - 1, τ_bad, h, C, α, f, g, heuristic))
    total_cost = positive + negative
    return total_cost
end

function find_optimal(T::Int64, τ::Float64, h::Float64,
    C::Array{Array{Float64, 1}, 1}, α::Float64, f, g)
    return 1/T * V(T, τ, h, C, α, f, g)
end

function random_heuristic(τ::Float64, h::Float64, Cₜ::Array{Float64, 1},
     α::Float64, f, g)
    return rand(1:length(Cₜ))
end

function neutral_heuristic(τ::Float64, h::Float64, Cₜ::Array{Float64, 1},
     α::Float64, f, g)
     return argmin(Cₜ .* Cₜ)
end

function least_difference_heuristic(τ::Float64, h::Float64, Cₜ::Array{Float64, 1},
      α::Float64, f, g)
      diffs = h .- Cₜ
      return argmin(diffs .^ 2)
end

function biggest_difference_heuristic(τ::Float64, h::Float64, Cₜ::Array{Float64, 1},
      α::Float64, f, g)
      diffs = h .- Cₜ
      return argmax(diffs .^ 2)
end

function highest_prob_heuristic(τ::Float64, h::Float64, Cₜ::Array{Float64, 1},
      α::Float64, f, g)
      ps = map(x -> p(h, x), Cₜ)
      return argmax(ps)
end

function benefit_risk_heuristic(τ::Float64, h::Float64, Cₜ::Array{Float64, 1},
      α::Float64, f, g)
      function b_per_r(o)
          denom = g(τ, h, o)#p(h, o)
          numer = (1 - (α * h + (1 - α) * o)^2)
          return numer / denom
      end
      scores = map(b_per_r, Cₜ)
      return argmax(scores)
end


heuristics = Dict(
    #"random_heuristic" => random_heuristic,
    "neutral_heuristic" => neutral_heuristic,
    "least_difference_heuristic" => least_difference_heuristic,
    "biggest_difference_heuristic" => biggest_difference_heuristic,
    "benefit_risk_heuristic" => benefit_risk_heuristic,
    "highest_prob_heuristic" => highest_prob_heuristic
)

function find_heuristic(T::Int64, τ::Float64, h::Float64,
    C::Array{Array{Float64, 1}, 1}, α::Float64, f, g, heuristic)
    return 1/T * V(T, τ, h, C, α, f, g, heuristic)
end
