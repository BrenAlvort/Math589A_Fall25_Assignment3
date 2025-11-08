function [N,K,c,d,a,alpha,beta_sin] = parse_coef_generic(coef_in, varargin)
% PARSE_COEF_GENERIC Robustly parse many coefficient formats.
% Returns canonical N,K,c,d,a,alpha,beta_sin (column vectors).
% Accepts:
%   - numeric vec with extra args N,K: parse_coef_generic(vec,N,K)
%   - struct with fields N,K,vec,c,d,a,alpha,beta
%   - wrapper struct that contains .coef
%   - struct that contains some numeric field that looks like vec (pick largest)
% On failure c,d may be empty.

% defaults
N = 0; K = 0; c = []; d = []; a = zeros(0,1); alpha = zeros(0,1); beta_sin = zeros(0,1);

% unwrap wrapper 'best' that contains .coef
if isstruct(coef_in) && isfield(coef_in,'coef') && (~isfield(coef_in,'vec') && ~isfield(coef_in,'c'))
    coef_in = coef_in.coef;
end

% numeric vector case
if isnumeric(coef_in)
    if numel(varargin) < 2
        error('parse_coef_generic: numeric coef requires N and K as extra args.');
    end
    N = varargin{1}; K = varargin{2};
    tmp = coef_in(:);
    if numel(tmp) ~= 2 + N + 2*K
        error('parse_coef_generic: coef length mismatch with provided N,K');
    end
    c = tmp(1); d = tmp(2);
    if N>0, a = tmp(3:2+N); end
    if K>0
        alpha = tmp(2+N+1:2+N+K);
        beta_sin = tmp(2+N+K+1:2+N+2*K);
    end
    return;
end

% struct case: use fields if present
if isstruct(coef_in)
    if isfield(coef_in,'N'), N = coef_in.N; end
    if isfield(coef_in,'K'), K = coef_in.K; end

    if isfield(coef_in,'c') && isfield(coef_in,'d')
        c = coef_in.c; d = coef_in.d;
    end

    if isfield(coef_in,'a'), a = coef_in.a(:); end
    if isfield(coef_in,'alpha'), alpha = coef_in.alpha(:); end
    if isfield(coef_in,'beta'), beta_sin = coef_in.beta(:); end

    if isfield(coef_in,'vec') && ~isempty(coef_in.vec)
        tmp = coef_in.vec(:);
        p = numel(tmp);
        K = floor((p-2)/2);
        N = p - 2 - 2*K;
        c = tmp(1); d = tmp(2);
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end

    % fallback: find the largest numeric vector field and treat it as vec
    fn = fieldnames(coef_in);
    bestlen = 0; bestvec = [];
    for i=1:numel(fn)
        val = coef_in.(fn{i});
        if isnumeric(val) && numel(val) > bestlen
            bestlen = numel(val);
            bestvec = val(:);
        end
    end
    if bestlen >= 2
        tmp = bestvec; p = numel(tmp);
        K = floor((p-2)/2);
        N = p - 2 - 2*K;
        c = tmp(1); d = tmp(2);
        if N>0, a = tmp(3:2+N); end
        if K>0
            alpha = tmp(2+N+1:2+N+K);
            beta_sin = tmp(2+N+K+1:2+N+2*K);
        end
        return;
    end
end

% ensure column shapes
a = a(:); alpha = alpha(:); beta_sin = beta_sin(:);
end
