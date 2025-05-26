import json
import os
from datetime import datetime
from neat.reporting import BaseReporter
from neat.six_util import itervalues
from neat.math_util import mean, stdev


class DataReporter(BaseReporter):
    """Reporter that saves detailed generation data to JSON files."""
    
    def __init__(self, output_dir):
        """
        Initialize the DataReporter.
        
        Args:
            output_dir (str): Directory where output files will be saved
        """
        self.output_dir = output_dir
        self.run_data = {
            'start_time': datetime.now().isoformat(),
            'generations': [],
            'extinctions': [],
            'solutions_found': [],
            'stagnant_species': []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize generation data
        self.current_generation = None
        self.generation_start_time = None
        
    def start_generation(self, generation):
        """Called at the start of each generation."""
        self.current_generation = generation
        self.generation_start_time = datetime.now()
        
    def end_generation(self, config, population, species_set):
        """Called at the end of each generation."""
        if self.current_generation is None:
            return
            
        generation_end_time = datetime.now()
        generation_duration = (generation_end_time - self.generation_start_time).total_seconds()
        
        # Collect species data
        species_data = []
        for sid in species_set.species:
            species = species_set.species[sid]
            species_info = {
                'id': sid,
                'age': self.current_generation - species.created,
                'size': len(species.members),
                'fitness': species.fitness,
                'adjusted_fitness': species.adjusted_fitness,
                'stagnation': self.current_generation - species.last_improved,
                'created_generation': species.created,
                'last_improved': species.last_improved
            }
            species_data.append(species_info)
        
        generation_data = {
            'generation': self.current_generation,
            'start_time': self.generation_start_time.isoformat(),
            'end_time': generation_end_time.isoformat(),
            'duration_seconds': generation_duration,
            'population_size': len(population),
            'num_species': len(species_set.species),
            'species': species_data
        }
        
        self.run_data['generations'].append(generation_data)
        
        # Save generation data to individual file
        gen_filename = f"generation_{self.current_generation:04d}.json"
        gen_filepath = os.path.join(self.output_dir, gen_filename)
        with open(gen_filepath, 'w') as f:
            json.dump(generation_data, f, indent=2)
    
    def post_evaluate(self, config, population, species, best_genome):
        """Called after population evaluation."""
        if self.current_generation is None:
            return
            
        # Calculate fitness statistics
        fitnesses = [genome.fitness for genome in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        fit_max = max(fitnesses)
        fit_min = min(fitnesses)
        
        # Get best genome info
        best_species_id = species.get_species_id(best_genome.key)
        
        evaluation_data = {
            'generation': self.current_generation,
            'fitness_stats': {
                'mean': fit_mean,
                'std': fit_std,
                'max': fit_max,
                'min': fit_min
            },
            'best_genome': {
                'key': best_genome.key,
                'fitness': best_genome.fitness,
                'size': best_genome.size(),
                'species_id': best_species_id
            },
            'population_data': [
                {
                    'key': genome.key,
                    'fitness': genome.fitness,
                    'size': genome.size() if hasattr(genome, 'size') else None
                }
                for genome in itervalues(population)
            ]
        }
        
        # Update the current generation data with evaluation info
        if self.run_data['generations'] and self.run_data['generations'][-1]['generation'] == self.current_generation:
            self.run_data['generations'][-1]['evaluation'] = evaluation_data
        
        # Save evaluation data to separate file
        eval_filename = f"evaluation_{self.current_generation:04d}.json"
        eval_filepath = os.path.join(self.output_dir, eval_filename)
        with open(eval_filepath, 'w') as f:
            json.dump(evaluation_data, f, indent=2)
    
    def post_reproduction(self, config, population, species):
        """Called after reproduction."""
        reproduction_data = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'new_population_size': len(population),
            'species_count': len(species.species)
        }
        
        # Save reproduction data
        repro_filename = f"reproduction_{self.current_generation:04d}.json"
        repro_filepath = os.path.join(self.output_dir, repro_filename)
        with open(repro_filepath, 'w') as f:
            json.dump(reproduction_data, f, indent=2)
    
    def complete_extinction(self):
        """Called when all species go extinct."""
        extinction_data = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat()
        }
        self.run_data['extinctions'].append(extinction_data)
        
        # Save extinction event
        extinction_filename = f"extinction_{self.current_generation:04d}.json"
        extinction_filepath = os.path.join(self.output_dir, extinction_filename)
        with open(extinction_filepath, 'w') as f:
            json.dump(extinction_data, f, indent=2)
    
    def found_solution(self, config, generation, best):
        """Called when a solution is found."""
        solution_data = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'best_genome': {
                'key': best.key,
                'fitness': best.fitness,
                'size': best.size()
            }
        }
        self.run_data['solutions_found'].append(solution_data)
        
        # Save solution data
        solution_filename = f"solution_{generation:04d}.json"
        solution_filepath = os.path.join(self.output_dir, solution_filename)
        with open(solution_filepath, 'w') as f:
            json.dump(solution_data, f, indent=2)
    
    def species_stagnant(self, sid, species):
        """Called when a species becomes stagnant."""
        stagnant_data = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'species_id': sid,
            'species_size': len(species.members)
        }
        self.run_data['stagnant_species'].append(stagnant_data)
    
    def info(self, msg):
        """Called for general information messages."""
        info_data = {
            'generation': self.current_generation,
            'timestamp': datetime.now().isoformat(),
            'message': msg
        }
        
        # Append to info log file
        info_filepath = os.path.join(self.output_dir, "info_log.json")
        if os.path.exists(info_filepath):
            with open(info_filepath, 'r') as f:
                info_log = json.load(f)
        else:
            info_log = []
        
        info_log.append(info_data)
        
        with open(info_filepath, 'w') as f:
            json.dump(info_log, f, indent=2)
    
    def save_summary(self):
        """Save a summary of the entire run."""
        self.run_data['end_time'] = datetime.now().isoformat()
        
        # Save complete run summary
        summary_filepath = os.path.join(self.output_dir, "run_summary.json")
        with open(summary_filepath, 'w') as f:
            json.dump(self.run_data, f, indent=2)
        
        print(f"Data saved to: {self.output_dir}")