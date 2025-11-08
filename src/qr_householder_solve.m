function [beta, RSS, R] = qr_householder_solve(A, b)
% QR_HOUSEHOLDER_SOLVE Solve min ||A*beta - b||_2 using Householder QR
%   [beta,RSS,R] = qr_householder_solve(A,b)
% Returns estimated beta, residual sum of squares RSS, and R (n-by-n upper tri)
[m, n] = size(A);
if size(b,1) ~= m
    error('qr_householder_solve: dimension mismatch between A and b');
end

Rfull = A;
cfull = b;

for k = 1:n
    x = Rfull(k:m, k);
    if all(x == 0)
        v = zeros(length(x),1);
    else
        alpha = -sign(x(1)) * norm(x);
        e1 = zeros(length(x),1); e1(1) = 1;
        v = x - alpha*e1;
        vnorm = norm(v);
        if vnorm ~= 0
            v = v / vnorm;
        else
            v = zeros(size(v));
        end
    end
    if any(v)
        Rfull(k:m, k:n) = Rfull(k:m, k:n) - 2 * v * (v' * Rfull(k:m, k:n));
        cfull(k:m) = cfull(k:m) - 2 * v * (v' * cfull(k:m));
    end
end

R = Rfull(1:n, 1:n);
c = cfull(1:n);

% Back substitution for R*beta = c
beta = zeros(n,1);
for i = n:-1:1
    if abs(R(i,i)) < eps * norm(R,inf)
        % treat as zero (ill-conditioned) — set parameter to zero
        beta(i) = 0;
    else
        beta(i) = (c(i) - R(i, i+1:n) * beta(i+1:n)) / R(i,i);
    end
end

r = A * beta - b;
RSS = sum(r.^2);
end
