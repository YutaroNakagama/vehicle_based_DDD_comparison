# Recalculating filter indices and using ANFIS for Importance Degree (ID) calculation

# Step 1: Calculate filter indices as before
X_low = X_combined[y_combined == 0]
X_high = X_combined[y_combined == 1]

fisher_index = [(X_high[col].mean() - X_low[col].mean()) / (X_high[col].std()**2 + X_low[col].std()**2) for col in X_combined.columns]
correlation_index = [np.cov(X_combined[col], y_combined)[0, 1] / (X_combined[col].std() * y_combined.std()) for col in X_combined.columns]
t_test_index = [abs(ttest_ind(X_low[col], X_high[col], equal_var=False)[0]) for col in X_combined.columns]
mutual_info_index = mutual_info_classif(X_combined, y_combined, discrete_features=False)

# Combine the filter indices into a DataFrame
filter_indices = pd.DataFrame({
    'Feature': X_combined.columns,
    'Fisher Index': fisher_index,
    'Correlation Index': correlation_index,
    'T-test Index': t_test_index,
    'Mutual Information Index': mutual_info_index
})

# Define a new ANFIS model for Importance Degree calculation
class ANFISForID:
    def __init__(self, num_rules=5):
        # Initializing parameters for ANFIS with num_rules rules
        self.num_rules = num_rules
        self.means = np.random.uniform(0, 1, (num_rules, 4))  # 4 inputs (filter indices)
        self.sigmas = np.random.uniform(0.1, 0.5, (num_rules, 4))
        self.consequents = np.random.uniform(0, 1, num_rules)  # Output importance degree (ID) range [0, 1]

    def forward(self, inputs):
        # Calculate firing strengths for each rule
        firing_strengths = np.zeros(self.num_rules)
        for i in range(self.num_rules):
            mu_values = [gaussian_mf(inputs[j], self.means[i, j], self.sigmas[i, j]) for j in range(4)]
            firing_strengths[i] = np.prod(mu_values)  # AND operation by multiplying membership values

        # Normalize firing strengths
        if firing_strengths.sum() == 0:
            return 0
        normalized_strengths = firing_strengths / firing_strengths.sum()

        # Calculate the output importance degree (ID) as a weighted sum of consequents
        importance_degree = np.dot(normalized_strengths, self.consequents)
        return importance_degree

    def set_parameters(self, means, sigmas, consequents):
        self.means = means
        self.sigmas = sigmas
        self.consequents = consequents

# Initialize the ANFIS model for ID calculation
anfis_id_model = ANFISForID(num_rules=5)

# Define PSO to optimize ANFIS parameters based on SVM accuracy as objective
def objective_function_pso(anfis, X_features, y_labels):
    # Calculate IDs for each feature using the current ANFIS parameters
    IDs = np.array([anfis.forward(row) for row in X_features])

    # Select features with ID >= 0.5
    selected_features = X_combined.loc[:, IDs >= 0.5]
    if selected_features.empty:
        return 1.0  # Penalize if no features are selected

    # Split data and scale for SVM
    X_train, X_test, y_train, y_test = train_test_split(selected_features, y_labels, test_size=0.2, random_state=42, stratify=y_labels)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM and calculate accuracy
    svm = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
    svm.fit(X_train_scaled, y_train)
    accuracy = svm.score(X_test_scaled, y_test)

    # Objective is to maximize accuracy (return negative accuracy for minimization)
    return -accuracy

# Initialize PSO particles for ANFIS parameter optimization
class Particle:
    def __init__(self, anfis):
        # Initialize positions and velocities for ANFIS parameters
        self.position_means = np.random.uniform(0, 1, anfis.means.shape)
        self.position_sigmas = np.random.uniform(0.1, 0.5, anfis.sigmas.shape)
        self.position_consequents = np.random.uniform(0, 1, anfis.consequents.shape)

        self.velocity_means = np.zeros_like(self.position_means)
        self.velocity_sigmas = np.zeros_like(self.position_sigmas)
        self.velocity_consequents = np.zeros_like(self.position_consequents)

        # Personal best
        self.best_position_means = np.copy(self.position_means)
        self.best_position_sigmas = np.copy(self.position_sigmas)
        self.best_position_consequents = np.copy(self.position_consequents)
        self.best_score = float('inf')

# PSO optimization parameters
c1, c2 = 2, 2
npop = 50  # Number of particles
w = 0.95
max_iter = 100

# Run PSO optimization
particles = [Particle(anfis_id_model) for _ in range(npop)]
global_best_score = float('inf')
global_best_means, global_best_sigmas, global_best_consequents = None, None, None

for iteration in range(max_iter):
    for particle in particles:
        # Set ANFIS parameters to current particle's position
        anfis_id_model.set_parameters(particle.position_means, particle.position_sigmas, particle.position_consequents)

        # Objective function (SVM accuracy based on selected features by ANFIS ID output)
        score = objective_function_pso(anfis_id_model, filter_indices[['Fisher Index', 'Correlation Index', 'T-test Index', 'Mutual Information Index']].values, y_combined)

        # Update personal and global bests
        if score < particle.best_score:
            particle.best_score = score
            particle.best_position_means = np.copy(particle.position_means)
            particle.best_position_sigmas = np.copy(particle.position_sigmas)
            particle.best_position_consequents = np.copy(particle.position_consequents)

        if score < global_best_score:
            global_best_score = score
            global_best_means = np.copy(particle.position_means)
            global_best_sigmas = np.copy(particle.position_sigmas)
            global_best_consequents = np.copy(particle.position_consequents)

    # Update particle velocities and positions
    for particle in particles:
        r1, r2 = np.random.rand(), np.random.rand()
        particle.velocity_means = w * particle.velocity_means + c1 * r1 * (particle.best_position_means - particle.position_means) + c2 * r2 * (global_best_means - particle.position_means)
        particle.velocity_sigmas = w * particle.velocity_sigmas + c1 * r1 * (particle.best_position_sigmas - particle.position_sigmas) + c2 * r2 * (global_best_sigmas - particle.position_sigmas)
        particle.velocity_consequents = w * particle.velocity_consequents + c1 * r1 * (particle.best_position_consequents - particle.position_consequents) + c2 * r2 * (global_best_consequents - particle.position_consequents)

        # Update positions
        particle.position_means += particle.velocity_means
        particle.position_sigmas += particle.velocity_sigmas
        particle.position_consequents += particle.velocity_consequents

# Set the best ANFIS parameters
anfis_id_model.set_parameters(global_best_means, global_best_sigmas, global_best_consequents)
best_accuracy = -global_best_score  # Convert back to positive accuracy for reporting

print(best_accuracy)

