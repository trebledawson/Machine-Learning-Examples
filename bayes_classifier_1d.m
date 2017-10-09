% ***************************** %
% Bayes Classifier in 1-D Data  %
% Glenn Dawson                  %
% 2017-09-21                    %
% ***************************** %

% Argument(s):
% S is the nth term of Z, where 1 < n < 1000
function P=bayes(S);

% Generate two 1-D data sets
MU1 = 0.5;              % Mean of class 1
SIGMA1 = 0.05;          % Stdev of class 1
MU2 = 0.6;              % Mean of class 2
SIGMA2 = 0.05;          % Stdev of class 2
omega1 = normrnd(MU1,SIGMA1,[1 1000]);
omega2 = normrnd(MU2,SIGMA2,[1 1000]);

% Plot the generated data histograms
histogram(omega1,100);
hold on;
histogram(omega2,100);
hold off;

% Calculate the mean and sigma of generated data
mu1 = mean(omega1);
mu2 = mean(omega2);
sigma1 = std(omega1);
sigma2 = std(omega2);

% Generate test data from originally defined normal distributions
Z = [];
for i=1:1000
    Zomega1 = normrnd(MU1,SIGMA1);
    Zomega2 = normrnd(MU2,SIGMA2);
    switch randi(2)
        case 1
            Z = [Z Zomega1];
        case 2
            Z = [Z Zomega2];
    end
end

% Plot test data histogram
hold on
histogram(Z,100);
hold off;

% Calculate P(Z|omegaj)
G1 = normpdf(Z(S),mu1,sigma1);
G2 = normpdf(Z(S),mu2,sigma2);

% Given P(Z(S))=1 and P(omegaj)=0.5, calculate Bayesian probability density
P1 = (G1*0.5)/1;
P2 = (G2*0.5)/1;

% Choose greatest Bayesian probability density
% Return [nth element of Z ; P(omega1|Z) ; P(omega2|Z) ; Classification]
switch max([P1 P2])
    case P1
        P=[Z(S); P1; P2; 1];
    case P2
        P=[Z(S); P1; P2; 2];
end
end