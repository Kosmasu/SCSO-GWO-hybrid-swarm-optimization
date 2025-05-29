import json
import os
import csv
from datetime import datetime
from neat.reporting import BaseReporter
from neat.six_util import itervalues
from neat.math_util import mean, stdev




class DataReporter(BaseReporter):
    """Clean reporter that saves essential training data in organized format."""
    
    def __init__(self, output_dir, config_file_path=None, continue_from_checkpoint=False):
        self.output_dir = output_dir
        self.config_file_path = config_file_path
        self.continue_from_checkpoint = continue_from_checkpoint
        
        # Create organized subdirectories
        self.data_dir = os.path.join(output_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize data storage
        self.generation_data = []
        self.best_fitness_history = []
        self.species_history = []
        
        # CSV files for easy analysis
        self.fitness_csv = os.path.join(self.data_dir, "fitness_history.csv")
        self.species_csv = os.path.join(self.data_dir, "species_history.csv")
        
        # Handle checkpoint continuation
        if continue_from_checkpoint:
            self._load_existing_data()
        else:
            self.run_start_time = datetime.now()
            # Copy NEAT config file to data directory
            if config_file_path and os.path.exists(config_file_path):
                import shutil
                config_dest = os.path.join(self.data_dir, "neat_config.txt")
                shutil.copy2(config_file_path, config_dest)
                print(f"📄 NEAT config saved to: {config_dest}")
            
            # Initialize CSV files
            self._init_csv_files()
    
    def _load_existing_data(self):
        """Load existing data when continuing from checkpoint."""
        checkpoint_file = os.path.join(self.data_dir, "training_checkpoint.json")
        
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.generation_data = checkpoint_data.get('generations', [])
                self.best_fitness_history = checkpoint_data.get('best_fitness_history', [])
                
                # Restore original start time
                start_time_str = checkpoint_data.get('run_info', {}).get('start_time')
                if start_time_str:
                    self.run_start_time = datetime.fromisoformat(start_time_str)
                else:
                    self.run_start_time = datetime.now()
                
                print(f"📊 Loaded checkpoint: {len(self.generation_data)} generations")
                print(f"   Best fitness so far: {max(self.best_fitness_history) if self.best_fitness_history else 0:.1f}")
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Warning: Could not load checkpoint data: {e}")
                print("   Starting fresh data collection...")
                self.run_start_time = datetime.now()
                self._init_csv_files()
        else:
            print("📊 No existing checkpoint data found, starting fresh...")
            self.run_start_time = datetime.now()
            self._init_csv_files()
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Fitness history CSV
        with open(self.fitness_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'timestamp', 'duration_seconds',
                'best_fitness', 'mean_fitness', 'std_fitness', 
                'min_fitness', 'max_fitness', 'population_size', 'num_species'
            ])
        
        # Species history CSV
        with open(self.species_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'generation', 'species_id', 'size', 'fitness', 
                'adjusted_fitness', 'age', 'stagnation'
            ])
    
    def start_generation(self, generation):
        """Called at the start of each generation."""
        self.current_generation = generation
        self.generation_start_time = datetime.now()
        
    def post_evaluate(self, config, population, species_set, best_genome):
        """Called after population evaluation - main data collection point."""
        if not hasattr(self, 'current_generation'):
            return
        
        # Skip if we already have data for this generation (checkpoint continuation)
        if any(gen_data['generation'] == self.current_generation for gen_data in self.generation_data):
            return
            
        generation_end_time = datetime.now()
        duration = (generation_end_time - self.generation_start_time).total_seconds()
        
        # Calculate fitness statistics
        fitnesses = [genome.fitness for genome in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        fit_max = max(fitnesses)
        fit_min = min(fitnesses)
        
        # Store generation data
        gen_data = {
            'generation': self.current_generation,
            'timestamp': generation_end_time.isoformat(),
            'duration_seconds': round(duration, 2),
            'fitness_stats': {
                'best': best_genome.fitness,
                'mean': round(fit_mean, 2),
                'std': round(fit_std, 2),
                'min': fit_min,
                'max': fit_max
            },
            'population_size': len(population),
            'num_species': len(species_set.species),
            'best_genome': {
                'key': best_genome.key,
                'fitness': best_genome.fitness,
                'nodes': len(best_genome.nodes),
                'connections': len(best_genome.connections)
            }
        }
        
        self.generation_data.append(gen_data)
        self.best_fitness_history.append(best_genome.fitness)
        
        # Write to fitness CSV (append mode for checkpoints)
        with open(self.fitness_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_generation, generation_end_time.isoformat(), round(duration, 2),
                best_genome.fitness, round(fit_mean, 2), round(fit_std, 2),
                fit_min, fit_max, len(population), len(species_set.species)
            ])
        
        # Write species data to CSV (append mode for checkpoints)
        with open(self.species_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for sid, species in species_set.species.items():
                writer.writerow([
                    self.current_generation, sid, len(species.members),
                    species.fitness, species.adjusted_fitness,
                    self.current_generation - species.created,
                    self.current_generation - species.last_improved
                ])
        
        # Save checkpoint every 10 generations
        if self.current_generation % 10 == 0:
            self._save_checkpoint()
        
        # Print progress
        print(f"Gen {self.current_generation:3d} | "
              f"Best: {best_genome.fitness:7.1f} | "
              f"Mean: {fit_mean:7.1f} | "
              f"Species: {len(species_set.species):2d} | "
              f"Time: {duration:5.1f}s")
    
    def _save_checkpoint(self):
        """Save current data as checkpoint."""
        checkpoint_data = {
            'run_info': {
                'start_time': self.run_start_time.isoformat(),
                'last_update': datetime.now().isoformat(),
                'total_generations': len(self.generation_data),
                'continue_from_checkpoint': self.continue_from_checkpoint
            },
            'generations': self.generation_data,
            'best_fitness_history': self.best_fitness_history
        }
        
        checkpoint_file = os.path.join(self.data_dir, "training_checkpoint.json")
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def found_solution(self, config, generation, best):
        """Called when solution is found."""
        solution_data = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "fitness": best.fitness,
            "nodes": len(best.nodes),
            "connections": len(best.connections),
        }

        solution_file = os.path.join(self.data_dir, "solution.json")
        with open(solution_file, "w") as f:
            json.dump(solution_data, f, indent=2)

        print(f"\n🎉 SOLUTION FOUND at generation {generation}!")
        print(f"   Fitness: {best.fitness:.1f}")

    def complete_extinction(self):
        """Called when all species go extinct."""
        extinction_data = {
            "generation": self.current_generation,
            "timestamp": datetime.now().isoformat(),
        }

        extinction_file = os.path.join(self.data_dir, "extinction.json")
        with open(extinction_file, "w") as f:
            json.dump(extinction_data, f, indent=2)

        print(f"\n💀 COMPLETE EXTINCTION at generation {self.current_generation}")

    def save_final_summary(self):
        """Save final training summary."""
        end_time = datetime.now()
        total_duration = (end_time - self.run_start_time).total_seconds()

        summary = {
            "run_info": {
                "start_time": self.run_start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": round(total_duration, 2),
                "total_duration_hours": round(total_duration / 3600, 2),
                "total_generations": len(self.generation_data),
            },
            "performance": {
                "best_fitness_ever": max(self.best_fitness_history)
                if self.best_fitness_history
                else 0,
                "final_best_fitness": self.best_fitness_history[-1]
                if self.best_fitness_history
                else 0,
                "generations_to_best": self.best_fitness_history.index(
                    max(self.best_fitness_history)
                )
                + 1
                if self.best_fitness_history
                else 0,
            },
            "files_created": {
                "fitness_history": "data/fitness_history.csv",
                "species_history": "data/species_history.csv",
                "training_checkpoint": "data/training_checkpoint.json",
                "final_summary": "data/final_summary.json",
            },
        }

        # Save final checkpoint
        self._save_checkpoint()

        # Save summary
        summary_file = os.path.join(self.data_dir, "final_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print(f"\n📊 TRAINING SUMMARY")
        print(
            f"   Duration: {total_duration / 3600:.1f} hours ({len(self.generation_data)} generations)"
        )
        print(f"   Best fitness: {max(self.best_fitness_history):.1f}")
        print(f"   Data saved to: {self.data_dir}")
