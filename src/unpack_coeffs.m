function coef = unpack_coeffs(beta, N, K)
    coef.c = beta(1);
    coef.d = beta(2);
    coef.a = beta(3:2+N);
    coef.alpha = beta(3+N : 2+N+K);
    coef.beta  = beta(3+N+K : 2+N+2*K);
end
