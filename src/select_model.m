function best = select_model(y, s, Ngrid, Kgrid, criterion)
% SELECT_MODEL search over (Ngrid, Kgrid) and return best struct with:
%   best.coef (struct with N,K,vec,c,d,a,alpha,beta), best.score, best.RSS, best.M, best.p ...
    if nargin < 5, criterion = 'bic'; end
    best = struct('score', inf, 'N', [], 'K', [], 'coef', [], 'RSS', [], 'M', [], 'p', [], 'S', []);
    T = numel(y);
    Sgrid = nan(numel(Ngrid), numel(Kgrid));

    for iN = 1:numel(Ngrid)
        N = Ngrid(iN);
        for jK = 1:numel(Kgrid)
            K = Kgrid(jK);
            try
                [A, b, meta] = build_design(y, s, N, K);
            catch
                Sgrid(iN, jK) = NaN;
                continue;
            end

            try
                [beta, ~, ~] = qr_solve_dense(A, b);
            catch ME
                % solver failure: skip candidate
                Sgrid(iN, jK) = NaN;
                continue;
            end

            p = 2 + N + 2*K;
            M = meta.rows;
            res = A*beta - b;
            RSS = res.'*res;
            if strcmpi(criterion,'bic')
                if RSS <= 0, RSS = max(RSS, eps); end
                S = M*log(RSS/M) + p*log(M);
            else
                if RSS <= 0, RSS = max(RSS, eps); end
                S = M*log(RSS/M) + 2*p;
            end
            Sgrid(iN, jK) = S;

            if S < best.score
                coef = unpack_coeffs(beta, N, K); % includes vec,N,K,c,d,a,alpha,beta
                best.score = S;
                best.N = N;
                best.K = K;
                best.coef = coef;
                best.RSS = RSS;
                best.M = M;
                best.p = p;
            end
        end
    end
    best.S = Sgrid;
    best.Ngrid = Ngrid;
    best.Kgrid = Kgrid;
end
