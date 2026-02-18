module L96SchneiderTopology

export mod1idx,
       linidx,
       jk_from_lin,
       wrap_lin,
       fast_neighbor_jk,
       validate_twisted_mapping

@inline mod1idx(i::Int, n::Int) = mod(i - 1, n) + 1

@inline linidx(j::Int, k::Int, J::Int) = (k - 1) * J + j

@inline function jk_from_lin(ell::Int, J::Int)
    kk = (ell - 1) ÷ J + 1
    jj = (ell - 1) % J + 1
    return jj, kk
end

@inline wrap_lin(ell::Int, K::Int, J::Int) = mod1idx(ell, K * J)

@inline function fast_neighbor_jk(j::Int, k::Int, offset::Int, K::Int, J::Int)
    ell = linidx(j, k, J)
    return jk_from_lin(wrap_lin(ell + offset, K, J), J)
end

function validate_twisted_mapping(K::Int, J::Int)
    j1, k1 = fast_neighbor_jk(J, 1, 1, K, J)
    (j1 == 1 && k1 == mod1idx(2, K)) || return false
    k1 != 1 || return false

    j2, k2 = fast_neighbor_jk(J, K, 1, K, J)
    (j2 == 1 && k2 == 1) || return false
    return true
end

end # module
