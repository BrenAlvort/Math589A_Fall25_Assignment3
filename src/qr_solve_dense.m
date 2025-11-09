function x = qr_solve_dense(A,b)
% Solve min ||A*x - b||_2 via explicit Householder QR (no normal equations)

    [m,n] = size(A);
    if size(b,1) ~= m || size(b,2) ~= 1
        error('Dimension mismatch: A is %dx%d, b is %dx%d.', m,n, size(b,1), size(b,2));
    end
    if m < n
        error('Underdetermined: m=%d must be >= n=%d.', m, n);
    end

    R = A;
    y = b;

    for k = 1:n
        xk = R(k:end, k);
        tau = norm(xk);
        if tau == 0
            % Column already zero below diagonal
            continue
        end
        % Stable sign choice: +1 if xk(1) >= 0, else -1
        if xk(1) >= 0
            sgn = 1;
        else
            sgn = -1;
        end
        alpha = -sgn * tau;          % so xk(1) - alpha is large in magnitude
        v = xk;
        v(1) = v(1) - alpha;
        vnorm = norm(v);
        if vnorm == 0
            % Degenerate reflector, skip
            continue
        end
        v = v / vnorm;

        % Apply reflector to trailing block and RHS
        R(k:end, k:end) = R(k:end, k:end) - 2 * (v * (v.' * R(k:end, k:end)));
        y(k:end)        = y(k:end)        - 2 *  v * (v.' * y(k:end));
    end

    % Extract upper triangular and Q^T b
    R = triu(R(1:n,1:n));
    c = y(1:n);

    % Back substitution with rank check
    x = zeros(n,1);
    for i = n:-1:1
        rii = R(i,i);
        if rii == 0
            error('Rank deficiency at R(%d,%d)=0.', i, i);
        end
        x(i) = (c(i) - R(i,i+1:end) * x(i+1:end)) / rii;
    end
end
