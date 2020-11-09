function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

Z_2 = X * Theta1';
A_2 = sigmoid(Z_2);
A_2 = [ones(m, 1) A_2];
Z_3 = A_2 * Theta2';
A_3 = sigmoid(Z_3);
cost = 0;
for i=1:m
	y_alt = zeros(num_labels, 1);
	y_alt(y(i)) = 1; % example: [1 0 0 0 0 0 ]
	y_alt_comp = 1 - y_alt; % y_alt_comp [0 1 1 1 1 1]
	cost = cost + log(A_3(i,:))*y_alt + log(1-A_3(i,:))*y_alt_comp;
end

regularization_factor = 0;
if(lambda != 0)
	% Removing biased term from regularization calculation.
 	theta1 = Theta1;
 	theta2 = Theta2;
	theta1(:, 1) = 0;
	theta2(:, 1) = 0;
	regularization_factor = (lambda/(2*m))*(sum(sum(theta1 .^ 2)')+ sum(sum(theta2 .^ 2)'));
endif
J = (-1)*cost/m + regularization_factor;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
% 

for i=1:m  
	%calculation of delta of the layer for a data set
	y_output = zeros(num_labels, 1);
	y_output(y(i), 1) = 1;
	output_layer_delta = A_3(i,:)' - y_output;
	hidden_layer_delta = (Theta2' * output_layer_delta) .* A_2(i,:)' .* (1 - A_2(i,:)');
	hidden_layer_delta = hidden_layer_delta(2:end);

	%calculation of theta gradients 

	Theta2_grad = (Theta2_grad +  output_layer_delta*A_2(i,:));

	Theta1_grad = (Theta1_grad + hidden_layer_delta*X(i,:));
end;

Theta2_grad = (1/m) * Theta2_grad; 
Theta1_grad = (1/m) * Theta1_grad; 


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m)*Theta2(:, 2:end);
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m)*Theta1(:, 2:end);














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
