struct ShellKinematics{T}
    a1::Vec{3,T}
    a2::Vec{3,T}
    A_metric::SymmetricTensor{2,2,T}
    a_metric::SymmetricTensor{2,2,T}
    E::SymmetricTensor{2,2,T}
    K::SymmetricTensor{2,2,T}
    γ::Vec{2,T}
end