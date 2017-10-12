% ********************************************* %
% Perceptron Discriminant with Gradient Descent %
% Glenn Dawson                                  %
% 2017-10-03                                    %
% ********************************************* %
clear all; clc

% ################################################################### %
%                                MAIN                                 %
% ################################################################### %
tic

% Generate three linearly separable distributions
cases = 1000;
omega1 = mvnrnd([1.2,3.25],[0.1 0;0 0.1],cases);
omega2 = mvnrnd([3.5,3.75],[0.1 0;0 0.1],cases);
omega3 = mvnrnd([2.75,1.5],[0.1 0;0 0.1],cases);
% omega4 = mvnrnd([5.1,2],[0.1 0;0 0.1],cases);
omega = [omega1;omega2;omega3];
% omega = [omega1;omega2;omega3;omega4];
label1 (1:cases) = 1;
label2 (1:cases) = 2;
label3 (1:cases) = 3;
% label4 (1:cases) = 4;
labels = [label1,label2,label3];
% labels = [label1,label2,label3,label4];

subplot(221)
plot(omega1(:,1),omega1(:,2),'b.')
title('Randomly Generated, Linearly-Separable Data')
hold on
grid
plot(omega2(:,1),omega2(:,2),'r.')
plot(omega3(:,1),omega3(:,2),'g.')
% plot(omega4(:,1),omega4(:,2),'m.')

% Plot perceptron classification
subplot(222)
class = simple_perceptron(omega,omega,labels);
title('Simple Perceptron Classification of Data')
hold on;

for i=1:length(class)
    if class(i) == 1
        plot(omega(i,1),omega(i,2),'b.')
    elseif class(i) == 2
        plot(omega(i,1),omega(i,2),'r.')
    elseif class(i) == 3
        plot(omega(i,1),omega(i,2),'g.')
%     elseif class(i) == 4
%         plot(omega(i,1),omega(i,2),'m.')
    else
        plot(omega(i,1),omega(i,2),'k.')
    end
end
grid;

% Plot decision boundary map
subplot(224)
ZX = 0:0.025:4.5;
ZY = 0:0.025:5;
% ZX = 0:0.025:7;
% ZY = 0:0.025:5;

Z = zeros(length(ZX)*length(ZY),2);
for j=1:length(ZX)
    for k=1:length(ZY)
        var = ((j-1)*length(ZY) + k);
        Z(var,1) = ZX(j);
        Z(var,2) = ZY(k);
    end
end

Zclass = simple_perceptron(Z,omega,labels);
title('Simple Perceptron Decision Boundary Map')
hold on;

for i=1:length(Zclass)
    if Zclass(i) == 1
        plot(Z(i,1),Z(i,2),'b.')
    elseif Zclass(i) == 2
        plot(Z(i,1),Z(i,2),'r.')
    elseif Zclass(i) == 3
        plot(Z(i,1),Z(i,2),'g.')
    elseif Zclass(i) == 4
        plot(Z(i,1),Z(i,2),'m.')
    else
        plot(Z(i,1),Z(i,2),'k.')
    end
end
grid;

% Calculate error rate
K = 10;
indices = crossvalind('Kfold',labels,K);
cp = classperf(labels);
for i = 1:K
    test = (indices == i); train = ~test;
    class = simple_perceptron(omega(test,:),omega(train,:),labels(train));
    classperf(cp,class,test);
end
fprintf('The k-fold cross-validated error rate is : %d.\n', cp.ErrorRate);

toc
% ################################################################### %
%                            END OF MAIN                              %
% ################################################################### %

% *******************************************************************
% Simple perceptron classifier function
% ----------
% Arguments:
% omega is a set of data that 
% *******************************************************************

function class = simple_perceptron(omega_test,omega_train,labels)
% Re-sort training data
omega1 = omega_train((labels==1),:);
omega1NOT = omega_train((labels~=1),:);
omega2 = omega_train((labels==2),:);
omega2NOT = omega_train((labels~=2),:);
omega3 = omega_train((labels==3),:);
omega3NOT = omega_train((labels~=3),:);
% omega4 = omega_train((labels==4),:);
% omega4NOT = omega_train((labels~=4),:);

% Calculate decision boundaries with gradient descent
W1 = gradient_descent(omega1,omega1NOT);
W2 = gradient_descent(omega2,omega2NOT);
W3 = gradient_descent(omega3,omega3NOT);
% W4 = gradient_descent(omega4,omega4NOT);

% Use decision boundaries as weights for perceptron nodes and use node
% decisions to classify test instance
[instances, ~] = size(omega_test);
classifications = zeros(instances,1);
for i=1:instances
    G1 = node(omega_test(i,:),W1);
    G2 = node(omega_test(i,:),W2);
    G3 = node(omega_test(i,:),W3);
%     G4 = node(omega_test(i,:),W4);
    
    [~,index] = max([G1,G2,G3]);
%     [~,index] = max([G1,G2,G3,G4]);    
    classifications(i) = index;
end

class = classifications;
end

% *******************************************************************
% Gradient descent function to find decision boundary between any two
% classes
% ----------
% Arguments: 
% X1 and X2 are Nx2 matrices of training data
% *******************************************************************

function W = gradient_descent(X1, X2)
% Create vertical matrix of data
X = vertcat(X1, X2);

% Create vertical matrix of labels
Y1 (1:length(X1),1) = 1;
Y2 (1:length(X2),1) = -1;
Y = vertcat(Y1, Y2);

% Gradient descent algorithm
w = [1;0;0];
eta = 0.03;
delta_w = [1;0;0];
check = 1;
count = 0;
% while pdist([delta_w';0 0 0]) > 0.0001    
while check > 0.1  
    % Find set of misclassified samples
    Xm_IDX = [];
    for i=1:length(X)
        Xi = [1;X(i,1);X(i,2)];
        if Y(i)*(w'*Xi) < 0
            Xm_IDX = [Xm_IDX i];
        end
    end
    
    % Use misclassified samples to update w
    sum = [0;0;0];
    for i=1:length(Xm_IDX)
        sum = sum + Y(Xm_IDX(i))*[1;X(Xm_IDX(i),1);X(Xm_IDX(i),2)];
    end
    
    delta_w = eta*sum;
    w = w + delta_w;
    check = pdist([delta_w';0 0 0]);
    count = count + 1;
end
W = w;
end

% *******************************************************************
% Perceptron node function
% ----------
% Arguments:
% X is an 1x2 vector of input data
% W is a 3x1 vector of input weights
% A and B are the choices the node can make
% *******************************************************************

function Z = node(X,W)
Z = W(1)*1 + W(2)*X(1) + W(3)*X(2);
end