function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%================= Sigmoid Function =========================

linearHypo = theta' * X';
logHypo = sigmoid(linearHypo);

%====================== End =================================

%===================== Cost function Calculation ============

PosX = find(y==1);
NegX = find(y==0);

PosLog = log(logHypo(1, PosX));
NegLog = log(1 - logHypo(1, NegX));

summation = sum(PosLog) + sum(NegLog);
summation = (-1/m)*summation;

%====================== End =================================

%====================== Cost function factor addition =======

ThetaSquare = theta(2:size(theta)).^2;
ThetaSquareSum = sum(ThetaSquare);
Factor = (lambda/(2*m)) * ThetaSquareSum;

% =========================== End ===========================

% ========================= regularized cost function =======

J = summation + Factor;

% ============================ End ==========================

% ============================ Gradient =====================

HypoYGrad = (1/m)*((logHypo' - y)'*X);

%============================== End =========================

%============================== Gradient factor addition ====

GradFactor = (lambda/m)*(theta);
GradFactor(1, 1) = 0;

%============================== End =========================

%============================== regularised Gradient ========

grad = HypoYGrad' + GradFactor;
end
