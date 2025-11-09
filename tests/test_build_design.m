% scripts/test_build_design_improved.m
% Improved unit-like test for build_design.m

% make sure src is on the path (adjust repo_root if needed)
repo_root = fileparts(fileparts(mfilename('fullpath'))); % if running from scripts/
if isempty(repo_root)
    % fallback if mfilename can't infer (run interactively)
    repo_root = 'C:\Users\hylia\Desktop\1st year\589\Math589A_Fall25_Assignment3';
end
addpath(fullfile(repo_root,'src'));
rehash toolboxcache; clear functions;

try
    % test parameters
    y = (1:20).';    % column vector
    s = 12;
    N = 3;
    K = 2;
    tol = 1e-12;     % tolerance for floating comparisons

    % call the function under test
    [A,b,meta] = build_design(y, s, N, K);

    % ---- basic shape checks ----
    assert( ismatrix(A) && isvector(b), 'build_design:outputType', 'A must be matrix and b must be vector.' );
    assert( size(A,1) == numel(y) - N, 'build_design:rows', ...
        'Expected %d rows in A (T-N) but found %d.', numel(y)-N, size(A,1) );
    expected_cols = 2 + N + 2*K;
    assert( size(A,2) == expected_cols, 'build_design:cols', ...
        'Expected %d columns in A but found %d.', expected_cols, size(A,2) );

    % response vector b starts at y_{N+1}
    assert( numel(b) == numel(y)-N, 'build_design:bsize', 'b has incorrect length.' );
    assert( all(b(1:min(end,3)) == y(N+1 : N+min(3,end))), 'build_design:bcontent', ...
        'First entries of b do not match y(N+1:N+3).' );

    % ---- check first row content (t = N+1) ----
    t1 = meta.t(1);
    row = A(1,:);
    % intercept
    assert( row(1) == 1, 'build_design:intercept', 'Intercept (col 1) must be 1.' );
    % trend (absolute t)
    assert( abs(row(2) - t1) <= tol, 'build_design:trend', 'Trend column (col 2) should equal absolute time t.' );

    % lags block indices: start at col 3, length N
    lag_start = 3; lag_end = 2 + N;
    lags = row(lag_start:lag_end);
    if N > 0
        expected_lags = zeros(1,N);
        for i = 1:N
            expected_lags(i) = y(t1 - i);
        end
        assert( all(abs(lags - expected_lags) <= tol), 'build_design:lags', ...
            'Lag columns do not match expected y(t-1),...,y(t-N).' );
    else
        assert( isempty(lags), 'build_design:lags_zeroN', 'There should be no lag columns when N==0.' );
    end

    % seasonal blocks
    cos_start = lag_end + 1;
    cos_end = cos_start + max(0,K)-1;
    sin_start = cos_end + 1;
    sin_end = sin_start + max(0,K)-1;

    if K > 0
        cos_blk = row(cos_start:cos_end);
        sin_blk = row(sin_start:sin_end);
        % compute expected seasonal values
        expected_cos = arrayfun(@(k) cos(2*pi*k*t1/s), 1:K);
        expected_sin = arrayfun(@(k) sin(2*pi*k*t1/s), 1:K);
        assert( all(abs(cos_blk - expected_cos) <= tol), 'build_design:cos', ...
            'Cosine seasonal columns mismatch.' );
        assert( all(abs(sin_blk - expected_sin) <= tol), 'build_design:sin', ...
            'Sine seasonal columns mismatch.' );
    else
        % ensure no seasonal columns
        assert( cos_start > size(A,2), 'build_design:cos_zeroK', 'Unexpected cosine columns when K==0.' );
    end

    % extra sanity: metadata
    assert( isstruct(meta) && isfield(meta,'rows') && isfield(meta,'p') && isfield(meta,'t'), ...
        'build_design:meta', 'meta struct is missing required fields.' );
    assert( meta.rows == size(A,1) && meta.p == size(A,2), 'build_design:meta_values', ...
        'meta.rows/meta.p do not match A size.' );

    fprintf('build_design OK — passed all checks (N=%d, K=%d, T=%d).\n', N, K, numel(y));
catch ME
    fprintf('build_design test FAILED: %s\n', ME.message);
    if isfield(ME,'stack') && ~isempty(ME.stack)
        fprintf('  at %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
    rethrow(ME);
end
