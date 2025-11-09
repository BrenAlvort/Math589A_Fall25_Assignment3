function S = score_model(RSS, M, p, criterion)
    if nargin<4 || isempty(criterion), criterion = 'bic'; end
    switch lower(criterion)
        case 'bic'
            if RSS <= 0, RSS = max(RSS, eps); end
            S = M*log(RSS/M) + p*log(M);
        case 'aic'
            if RSS <= 0, RSS = max(RSS, eps); end
            S = M*log(RSS/M) + 2*p;
        otherwise
            error('Unknown criterion');
    end
end
