% ***************************************** %
% Linear Regression using Gradient Descent  %
% Glenn Dawson                              %
% 2017-09-15                                %
% ***************************************** %
theta = [0;0];
alpha = 0.0001;
gradient_descent(theta,alpha);

% Arguments: 1x2 vector theta, scalar alpha
function m=gradient_descent(theta,alpha)

% Random data generation
data_b = 7;                         % Vertical displacement of random data
data_m = 2;                         % Slope of random data
n = 100;                            % Number of random samples
noise = rand(n,1);                  % Data noise
x = rand(n,1).*10;                  % Random x values
y = data_b + data_m*x + noise;      % Corresponding y data

% Gradient Descent Algorithm
cost = 1;                           % Initialize cost variable
costs = [];                         % Initialize costs for plot
while abs(cost) > 0.0001            % Minimize cost function
    sum = [0 ; 0];                  % Initialize sum of diff(H(theta))
    cost = 0;                       % Reset cost
    for i=1:length(x)               % Compute cost and sum of H and J
        X = [1;x(i)];
        sum = sum + (theta'*X - y(i))*X;
        cost = cost + (theta'*X - y(i)); 
    end
    
    costs = [costs abs(cost)];      % For plotting costs over iterations
    
    theta = theta - alpha*sum;      % Update theta
end

% Plot results
subplot(211)
plot(x,y,'.'); title('Plot of randomly generated data (with linear regression)')
hold on
plot(x,theta(2)*x+theta(1));
hold off

subplot(212)
plot(costs,'.'); title('Cost function values over gradient descent iterations');

% Return final values of theta
m=theta;
end