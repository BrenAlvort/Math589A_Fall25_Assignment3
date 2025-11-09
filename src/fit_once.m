function out = fit_once(y, s, N, K)
% FIT_ONCE  fit single (N,K) model and return results in struct
    [A,b,meta] = build_design(y, s, N, K);
    [beta, ~, ~] = qr_solve_dense(A, b);
    res  = A*beta - b;
    RSS  = res.' * res;
    coef = unpack_coeffs(beta, N, K);   % returns full struct with vec,N,K,c,d,a,alpha,beta
    out = struct('beta', beta, 'coef', coef, 'RSS', RSS, 'M', meta.rows, 'p', meta.p, ...
                 'N', N, 'K', K, 's', s, 'res', res);
end
