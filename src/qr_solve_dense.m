function x = qr_solve_dense(A,b)
% QR_SOLVE_DENSE  Solve min ||A*x - b||_2 using explicit Householder QR.
% No backslash, no normal equations, only dense Householder.
    [m,n] = size(A);
    R = A;
    y = b(:);

    for k = 1:n
        v = R(k:end,k);
        if all(v==0)
            continue;
        end
        alpha = -sign(v(1)) * norm(v);
        v(1) = v(1) - alpha;
        vnorm = norm(v);
        if vnorm == 0
            continue;
        end
        v = v / vnorm;
        R(k:end,k:end) = R(k:end,k:end) - 2*v*(v.'*R(k:end,k:end));
        y(k:end) = y(k:end) - 2*v*(v.'*y(k:end));
    end

    R = triu(R(1:n,1:n));
    x = zeros(n,1);
    for i = n:-1:1
        if abs(R(i,i)) < eps * norm(R,inf)
            x(i) = 0;
        else
            x(i) = (y(i) - R(i,i+1:end)*x(i+1:end)) / R(i,i);
        end
    end
end
