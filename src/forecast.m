function yF = forecast(y, s, coef, H)
y = y(:); T = numel(y); N = numel(coef.a); K = numel(coef.alpha);
M = T - N;
yF = zeros(H,1);
for h = 1:H
    t = T + h;                 % absolute time for season/lag
    r = M + h;                 % row-index for trend
    sea = 0;
    for k = 1:K
        sea = sea + coef.alpha(k)*cos(2*pi*k*t/s) + coef.beta(k)*sin(2*pi*k*t/s);
    end
    acc = coef.c + coef.d*r + sea;     % trend uses r
    for i = 1:N
        idx = t - i;
        if idx <= T
            acc = acc + coef.a(i) * y(idx);
        else
            acc = acc + coef.a(i) * yF(idx - T);
        end
    end
    yF(h) = acc;
end
end
