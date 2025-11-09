function [x, R, cHat] = qr_solve_dense(A,b)
% QR_SOLVE_DENSE  Solve min ||A*x - b||_2 via explicit Householder QR.
%   [x, R, cHat] = qr_solve_dense(A,b)
%   Returns:
%     x    - solution of least-squares (n x 1)
%     R    - upper triangular factor (n x n)
%     cHat - first n entries of Q'*b (used if caller needs residual)
%
    narginchk(2,2);
    [m,n] = size(A);
    if m < n
        error('Underdetermined system: number of rows m=%d must be >= n=%d.', m, n);
    end
    if ~isvector(b) || numel(b)~=m
        error('Dimension mismatch between A (%dx%d) and b (%dx%d).', m, n, numel(b), 1);
    end

    Rfull = A;
    cfull = b(:);

    for k = 1:n
        xk = Rfull(k:m,k);
        if all(xk == 0)
            continue;
        end
        alpha = -sign(xk(1)) * norm(xk);
        v = xk;
        v(1) = v(1) - alpha;
        vnorm = norm(v);
        if vnorm == 0
            continue;
        end
        v = v / vnorm;

        % apply to trailing block and RHS
        Rfull(k:m,k:n) = Rfull(k:m,k:n) - 2 * (v * (v' * Rfull(k:m,k:n)));
        cfull(k:m)      = cfull(k:m)      - 2 * (v * (v' * cfull(k:m)));
    end

    R = triu(Rfull(1:n,1:n));
    cHat = cfull(1:n);

    % back substitution with tolerance (avoid division by tiny diag)
    tol = eps * norm(R,inf) * n;
    x = zeros(n,1);
    for i = n:-1:1
        rii = R(i,i);
        if abs(rii) <= tol
            % treat as (nearly) zero pivot -> set parameter to zero (regularized)
            x(i) = 0;
        else
            x(i) = (cHat(i) - R(i,i+1:end)*x(i+1:end)) / rii;
        end
    end
end
