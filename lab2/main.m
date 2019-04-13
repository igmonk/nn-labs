% Initialization
clear ; close all; clc

x = -3:0.05:3;
y = cos(4 * pi * x); % [1 121]
k = 5;

[ignore, m] = size(y);

% Initialization of Training Set
ts = zeros(k, m); % [5 m]
n = k;

y_ext = [zeros(1, n), y];

for i = 1:m
  ts(:, i) = y_ext(i:i+n-1);
end

% Training
initial_theta = zeros(1, n + 1); % [1 n+1]
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
  fminunc(@(t)(costFunction(t, ts, y)), initial_theta, options);

disp(size(theta));
disp(theta);

p = predict(theta, ts);
plot(x, y, 'bx', 'LineWidth', 2, x, p, 'r');
pause;

% Loss calculation
loss = abs(y - p);

% Loss plot
figure;
hold on;
plot(x, loss, 'b');
hold off;

pause;
