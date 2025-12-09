class ProgressTracker:
    """Class to track progress for different stages of the analysis"""

    def __init__(self, status_element, progress_bar, progress_text, detail_element, log_function=None):
        """Initialize progress tracker with UI elements"""
        self.status_element = status_element
        self.progress_bar = progress_bar
        self.progress_text = progress_text
        self.detail_element = detail_element
        self.log_function = log_function
        self.training_history = []  # Store detailed training output

    def update(self, progress, message, details=""):
        """Update progress indicators"""
        self.status_element.markdown(
            '<span class="status-running">Running</span>', unsafe_allow_html=True)
        self.progress_bar.progress(progress)
        self.progress_text.markdown(f"**{message}**")
        self.detail_element.text(details)

        # Add log entry if log function is provided
        if self.log_function:
            stage_name = getattr(self, 'stage_name', 'Progress')
            self.log_function(f"{stage_name} ({progress:.0%}): {message} - {details}")

    def add_training_output(self, output):
        """Add detailed training output to history and update display"""
        # Initialize full history list if it doesn't exist
        if not hasattr(self, 'full_history'):
            self.full_history = []

        # Add new output to history
        self.full_history.append(output)

        # Format epoch headers to make them stand out
        formatted_output = output
        if "Epoch" in output and "Initial" not in output and "Finished" not in output:
            formatted_output = "\n" + "="*50 + "\n" + output
            self.full_history[-1] = formatted_output  # Replace the last entry with formatted version

        # Create a continuous display of all output
        # Limit the size to prevent excessive UI slowdown
        display_limit = 50  # Show last 50 entries
        display_history = self.full_history[-display_limit:] if len(self.full_history) > display_limit else self.full_history

        # Format as code block with monospace font for better readability
        self.detail_element.code("\n".join(display_history))

        # Also log it if log function is provided
        if self.log_function:
            stage_name = getattr(self, 'stage_name', 'Training')
            self.log_function(f"{stage_name}: {output}")

    def mark_complete(self, message="Complete"):
        """Mark this stage as complete"""
        self.status_element.markdown(
            f'<span class="status-success">{message}</span>', unsafe_allow_html=True)


def create_model_tracker(model_status, model_progress_bar, model_progress_text, model_detail, add_log):
    """Create a progress tracker for model loading"""
    tracker = ProgressTracker(model_status, model_progress_bar, model_progress_text, model_detail, add_log)
    tracker.stage_name = "Load Model"
    return tracker


def create_dataset_tracker(dataset_status, dataset_progress_bar, dataset_progress_text, dataset_detail, add_log):
    """Create a progress tracker for dataset loading"""
    tracker = ProgressTracker(dataset_status, dataset_progress_bar, dataset_progress_text, dataset_detail, add_log)
    tracker.stage_name = "Load Dataset"
    return tracker


def create_embedding_tracker(embedding_status, embedding_progress_bar, embedding_progress_text, embedding_detail, add_log):
    """Create a progress tracker for embedding creation"""
    tracker = ProgressTracker(embedding_status, embedding_progress_bar, embedding_progress_text, embedding_detail, add_log)
    tracker.stage_name = "Create Representations"
    return tracker


def create_training_tracker(training_status, training_progress_bar, training_progress_text, training_detail, add_log):
    """Create a progress tracker for probe training"""
    tracker = ProgressTracker(training_status, training_progress_bar, training_progress_text, training_detail, add_log)
    tracker.stage_name = "Train Probe"
    return tracker

# Create a custom print function that will also update UI
def create_ui_print_function(original_print, progress_tracker):
    """Create a custom print function that shows output in UI and console"""
    # Keep track of last update time to avoid too frequent UI refreshes
    import time
    last_update_time = [0]  # Using list for nonlocal access
    min_update_interval = 0.2  # Seconds between UI updates

    def ui_print(*args, **kwargs):
        # Call the original print function for console output
        original_print(*args, **kwargs)

        # Also update the UI if we have a valid progress tracker
        if progress_tracker and hasattr(progress_tracker, 'add_training_output'):
            # Convert args to string
            output_text = " ".join(map(str, args))

            # Only update UI at reasonable intervals for smooth display
            current_time = time.time()
            # Always update for important messages or if enough time has passed
            important_output = any(key in output_text for key in ["Initial", "Epoch", "Finished", "Layer", "TEST"])

            if important_output or (current_time - last_update_time[0] >= min_update_interval):
                progress_tracker.add_training_output(output_text)
                last_update_time[0] = current_time

    return ui_print

# Create a custom print function that only logs to console and log file, but not the UI detail
def create_console_print_function(original_print, log_function):
    """Create a custom print function that outputs to console and log file but not the UI detail pane"""

    def console_print(*args, **kwargs):
        # Call the original print function for console output
        original_print(*args, **kwargs)

        # Add to log file if we have a log function
        if log_function:
            # Convert args to string
            output_text = " ".join(map(str, args))
            log_function(output_text)

    return console_print