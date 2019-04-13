% Initialization
clear ; close all; clc

x = -9:0.1:9; % [1 181]
y = (1.7 * x - 1.9) .* cos(1.4 * x - 1.7); % [1 181]

[n m] = size(x);

plot(x, y, 'bx', 'LineWidth', 2);
pause;

in_layer_size = n;
hidden_layer_size = 10;
out_layer_size = 1;

initial_theta1 = randInitializeWeights(in_layer_size, hidden_layer_size); % [3 2]
initial_theta2 = randInitializeWeights(hidden_layer_size, out_layer_size); % [1 4]

% Unroll parameters
initial_nn_params = [initial_theta1(:) ; initial_theta2(:)];

% Training
X = x';
options = optimset('MaxIter', 50);
lambda = 1;
nnCostFunction = @(p) costFunction(
  p, ...
  in_layer_size, ...
  hidden_layer_size, ...
  out_layer_size, X, y, lambda);

[nn_params, cost] = fmincg(nnCostFunction, initial_nn_params, options);

% Obtain theta1 and theta2 back from nn_params
theta1 = reshape(nn_params(1:hidden_layer_size * (in_layer_size + 1)), ...
  hidden_layer_size, (in_layer_size + 1));

theta2 = reshape(nn_params((1 + (hidden_layer_size * (in_layer_size + 1))):end), ...
  out_layer_size, (hidden_layer_size + 1));

disp('th1:');
disp(size(theta1));
disp(theta1);

disp('th2:');
disp(size(theta2));
disp(theta2);

% Prediction
p = predict(theta1, theta2, X);

figure;
hold on;
plot(x, y, 'b', x, p', 'r');
hold off;

pause;

