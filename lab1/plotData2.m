function plotData2(X, y)

% Create New Figure
figure;
hold on;

% Find indices of 1 class examples
class1 = find(y == 1);

% Find indeces of 2 class examples
class2 = find(y == 2);

% Find indices of 3 class examples
class3 = find(y == 3);

% Find indeces of 4 class examples
class4 = find(y == 4);

% Plot examples
plot(X(class1, 1), X(class1, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(class2, 1), X(class2, 2), 'r+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(class3, 1), X(class3, 2), 'g+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(class4, 1), X(class4, 2), 'b+', 'LineWidth', 2, 'MarkerSize', 7);

hold off;

end
