
% ******************************************************************* %
% Cumulative Distribution Function of Arbitrary Discrete Distribution %
% Glenn Dawson                                                        %
% 2017-09-19                                                          %
% ******************************************************************* %

% Arguments:
% X is a vector containing normalized probabilities
% Y is an integer representing the number of samples to be drawn from X
% Returns:
% 1-by-Y vector m ~ X

function m=cumdistgen(X, Y)

% Generate a sample from the distribution
sample = [];
for i=1:Y
    r=rand;                         % Generate a random number {0 1}
    cumdist=0;                      % Reset cumulative distribution
    j=1;                            % Counter for X
    while r >= cumdist              
        cumdist = cumdist + X(j);   % Add X(j) probability to extant sum
        j = j+1;                    % Increment to X(j+1)
    end
    sample = [sample (j-1)];        % Add chosen element of X to sample
end

% Return sample
m=sample; 