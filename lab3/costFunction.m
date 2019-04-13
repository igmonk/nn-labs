function [J grad] = costFunction(
  nn_params, ...
  input_layer_size, ...
  hidden_layer_size, ...
  num_labels, ...
  X, y, lambda)

  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  Theta1 = reshape(
    nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(
    nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

  % Setup some useful variables
  m = size(X, 1);

  % Return the following variables correctly
  J = 0;
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));

  % Part 1 - Forward Propagation
  X = [ones(m, 1), X];
  a1 = X';
  z2 = Theta1 * a1;
  a2 = sigmoid(z2);
  a2 = [ones(1, m); a2];
  z3 = Theta2 * a2;
  a3 = z3;
  diff = a3 - y;
  square = (diff .^ 2);
  J = (0.5 / m) * sum(sum(square));

  % Regularization
  % Do not regularize the terms that correspond to the bias (1st column of each theta matrix)
  Theta1Reg = Theta1(1:end, 2:end); % exclude 1st column
  Theta2Reg = Theta2(1:end, 2:end); % exclude 1st column
  regTerm = (lambda / (2 * m)) * (sum(sum(Theta1Reg .^ 2)) + sum(sum(Theta2Reg .^ 2)));
  J = J + regTerm;

  % Part 2 - Backpropagation
  % size(X) = [m 401]
  % size(Theta1) = [25 401]
  % size(Theta2) = [11 26]

  Delta1 = zeros(size(Theta1));
  Delta2 = zeros(size(Theta2));

  for i=1:m

    % Forward propagation (J calculation code can be reused)
    a1 = X(i, :)';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    %a3 = sigmoid(z3);
    a3 = z3;

    % Backpropagation - error terms
    delta3 = a3 - y(i);
    delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
    delta2 = delta2(2:end); % Skip delta2[0]

    % Backpropagation - gradients
    Delta2 = Delta2 + delta3 * a2';
    Delta1 = Delta1 + delta2 * a1';
  endfor;

  Theta2_grad_reg = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2Reg];
  Theta1_grad_reg = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1Reg];

  Theta2_grad = (1 / m) * Delta2 + Theta2_grad_reg;
  Theta1_grad = (1 / m) * Delta1 + Theta1_grad_reg;

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
