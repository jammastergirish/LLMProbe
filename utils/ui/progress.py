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
        self.training_history.append(output)
        # Display the most recent outputs (up to last 10 lines)
        recent_history = self.training_history[-10:]

        # Format as code block with monospace font for better readability
        self.detail_element.markdown(f"```\n{'  \n'.join(recent_history)}\n```")

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


def create_autoencoder_tracker(autoencoder_status, autoencoder_progress_bar, autoencoder_progress_text, autoencoder_detail, add_log):
    """Create a progress tracker for sparse autoencoder training"""
    tracker = ProgressTracker(autoencoder_status, autoencoder_progress_bar, autoencoder_progress_text, autoencoder_detail, add_log)
    tracker.stage_name = "Train Sparse Autoencoder"
    return tracker

# Create a custom print function that will also update UI
def create_ui_print_function(original_print, progress_tracker):
    """Create a custom print function that shows output in UI and console"""
    def ui_print(*args, **kwargs):
        # Call the original print function for console output
        original_print(*args, **kwargs)

        # Also update the UI if we have a valid progress tracker
        if progress_tracker and hasattr(progress_tracker, 'add_training_output'):
            # Convert args to string
            output_text = " ".join(map(str, args))
            progress_tracker.add_training_output(output_text)

    return ui_print