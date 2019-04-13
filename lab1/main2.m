%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the x & y coordinates and the third column contains the label.
data = load('data3.txt');

X = data(:, [1, 2]); y = data(:, 3);
num_labels = 4;


%% ==================== Part 1: Plotting ====================
plotData2(X, y);

% Put some labels
hold on;
% Labels and Legend
xlabel('X axis')
ylabel('Y axis')

% Specified in plot order
legend('Class 0', 'Class 1', 'Class 2', 'Class 3')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============ Part 2: Compute Cost and Gradient ============

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term
X = [ones(m, 1) X]; % [12 3]

% Initialize fitting parameters
initial_theta = zeros(n + 1, num_labels); % [3 4]
iterations = 100; % [100 3]
alpha = 0.01; % [100 1]

% Recode labels as vectors containing values 0 or 1
yVec = zeros(m, num_labels);
yRange = [1:num_labels]'; % [4 1]
for i=1:m
  yVec(i, :) = (yRange == y(i));
endfor;

%yVec = [
%  0 0.001
%  0 0.001
%  0 0.001
%  0 1
%  0 1
%  0 1
%  1 0
%  1 0
%  1 0
%  1 1
%  1 1
%  1 1
%];

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = ...
  fminunc(@(t)(costFunction(t, X, yVec)), initial_theta, options);
%theta = gradientDescent(X, y, initial_theta, alpha, iterations);

disp('theta:');
disp(theta);

% Plot Boundary
plotDecisionBoundary2(theta, X, y);
pause;
