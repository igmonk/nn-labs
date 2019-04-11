%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the x & y coordinates and the third column contains the label.
data = load('data1.txt');

X = data(:, [1, 2]); y = data(:, 3);


%% ==================== Part 1: Plotting ====================
plotData(X, y);

% Put some labels
hold on;
% Labels and Legend
xlabel('X axis')
ylabel('Y axis')

% Specified in plot order
legend('Class 1', 'Class 0')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============ Part 2: Compute Cost and Gradient ============

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1); % [3 1]
iterations = 100; % [100 3]
alpha = 0.01; % [100 1]

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
  fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
%theta = gradientDescent(X, y, initial_theta, alpha, iterations);

disp('theta:');
disp(theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
pause;
