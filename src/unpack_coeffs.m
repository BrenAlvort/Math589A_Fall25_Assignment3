function coef = unpack_coeffs(beta, N, K)
% UNPACK_COEFFS  helper matching our indexing convention
beta = beta(:);
coef.c = beta(1);
coef.d = beta(2);
coef.a = zeros(max(0,N),1);
coef.alpha = zeros(max(0,K),1);
coef.beta = zeros(max(0,K),1);

idx = 3;
for i = 1:N
    coef.a(i) = beta(idx); idx = idx + 1;
end
for k = 1:K
    coef.alpha(k) = beta(idx); idx = idx + 1;
end
for k = 1:K
    coef.beta(k) = beta(idx); idx = idx + 1;
end
end
