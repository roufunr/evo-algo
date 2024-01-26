import random

class Particle:
    def __init__(self, dimension):
        self.position = [random.uniform(-5, 5) for _ in range(dimension)]
        self.velocity = [random.uniform(-1, 1) for _ in range(dimension)]
        self.best_position = self.position
        self.best_fitness = float('inf')

def objective_function(x):
    # Define your objective function here
    return sum([i**2 for i in x])

def update_velocity(particle, global_best_position, w=0.5, c1=1.5, c2=1.5):
    for i in range(len(particle.velocity)):
        r1, r2 = random.random(), random.random()
        cognitive_component = c1 * r1 * (particle.best_position[i] - particle.position[i])
        social_component = c2 * r2 * (global_best_position[i] - particle.position[i])
        particle.velocity[i] = w * particle.velocity[i] + cognitive_component + social_component

def update_position(particle):
    for i in range(len(particle.position)):
        particle.position[i] = particle.position[i] + particle.velocity[i]

def pso_algorithm(num_particles, num_dimensions, num_iterations):
    particles = [Particle(num_dimensions) for _ in range(num_particles)]

    global_best_particle = min(particles, key=lambda x: objective_function(x.position))
    global_best_position = global_best_particle.position

    for _ in range(num_iterations):
        for particle in particles:
            fitness = objective_function(particle.position)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            if fitness < objective_function(global_best_position):
                global_best_position = particle.position

        for particle in particles:
            update_velocity(particle, global_best_position)
            update_position(particle)

    return global_best_position, objective_function(global_best_position)

if __name__ == "__main__":
    num_particles = 20
    num_dimensions = 3
    num_iterations = 50

    best_position, best_fitness = pso_algorithm(num_particles, num_dimensions, num_iterations)

    print("Best Position:", best_position)
    print("Best Fitness:", best_fitness)
