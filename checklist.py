# =============================================================================
# CHECKLIST.PY - Modeling Checklist Logger
# =============================================================================
# This module tracks the progress of the AI stock prediction pipeline.
# It maintains a checklist of 7 critical steps and marks each as
# complete or failed as the workflow progresses.
#
# Student: 711524BCS164 | SL.NO: 54
# =============================================================================

from datetime import datetime  # For timestamping when each step completes


class ChecklistLogger:
    """
    A logger class that tracks the status of modeling workflow steps.
    
    This class maintains a checklist of predefined tasks that must be completed
    during the AI stock prediction pipeline. Each task can be marked as done,
    failed, or pending. The logger provides formatted output for monitoring progress.
    
    Attributes:
        items (list): List of checklist item descriptions
        status (dict): Dictionary mapping item names to their current status
        timestamps (dict): Dictionary mapping item names to completion timestamps
    """
    
    def __init__(self):
        """
        Initialize the checklist with 7 predefined modeling steps.
        
        All items start with status "pending" and will be updated
        as the pipeline executes successfully or encounters errors.
        """
        # === STEP 1: Define the 7 checklist items ===
        # These represent the critical phases of the ML pipeline
        self.items = [
            "Finance Dataset Downloaded & Verified",
            "Data Scaled with MinMaxScaler",
            "Sliding Window Sequences Created (60-day lookback)",
            "LSTM Model Architecture Defined",
            "Hyperparameters Configured (lr=0.001, epochs=50, batch=32)",
            "Model Trained with Early Stopping",
            "Evaluation Metrics Computed (RMSE, MAPE, R²)"
        ]
        
        # Initialize all statuses as "pending" (⏳)
        # Dictionary comprehension creates {item: "pending"} for each item
        self.status = {item: "pending" for item in self.items}
        
        # Store timestamps when items are marked complete
        # Helps with debugging and performance tracking
        self.timestamps = {}
    
    def mark_done(self, item: str) -> None:
        """
        Mark a specific checklist item as successfully completed.
        
        This method updates the status of an item to "done" and records
        the current timestamp. It also prints a confirmation message.
        
        Args:
            item (str): The exact name of the checklist item to mark as done.
                       Must match one of the predefined items in self.items.
        
        Example:
            >>> logger = ChecklistLogger()
            >>> logger.mark_done("Finance Dataset Downloaded & Verified")
            ✅ Done: Finance Dataset Downloaded & Verified
        """
        # Check if the item exists in our checklist (safety check)
        if item in self.status:
            # Update status to "done" (will display as ✅)
            self.status[item] = "done"
            # Record the exact time this step completed
            self.timestamps[item] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Print immediate feedback to console with emoji marker
            print(f"✅ Done: {item}")
    
    def mark_fail(self, item: str) -> None:
        """
        Mark a specific checklist item as failed.
        
        Use this method when a pipeline step encounters an error
        or cannot complete successfully. This helps identify where
        the workflow broke down.
        
        Args:
            item (str): The exact name of the checklist item to mark as failed.
        
        Example:
            >>> logger = ChecklistLogger()
            >>> logger.mark_fail("Model Trained with Early Stopping")
            ❌ Failed: Model Trained with Early Stopping
        """
        # Verify item exists before updating
        if item in self.status:
            # Set status to "fail" (will display as ❌)
            self.status[item] = "fail"
            # Record when the failure occurred
            self.timestamps[item] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Print error notification with emoji marker
            print(f"❌ Failed: {item}")
    
    def get_status(self) -> list:
        """
        Retrieve the current status of all checklist items.
        
        Returns:
            list: A list of tuples where each tuple contains:
                  (item_name: str, status: str)
                  Status will be "pending", "done", or "fail"
        
        Example:
            >>> logger = ChecklistLogger()
            >>> logger.mark_done("Finance Dataset Downloaded & Verified")
            >>> logger.get_status()
            [('Finance Dataset Downloaded & Verified', 'done'), (...)]
        """
        # Use list comprehension to create tuples of (item, status)
        # Maintains the original order of items
        return [(item, self.status[item]) for item in self.items]
    
    def print_summary(self) -> None:
        """
        Display a formatted summary of all checklist items and their statuses.
        
        This method prints a visual table showing:
        - Checkmark (✅) for completed items
        - Cross mark (❌) for failed items
        - Hourglass (⏳) for pending items
        - Item descriptions aligned in a clean format
        
        Useful for console output and debugging the pipeline.
        """
        # Print header with decorative border
        print("\n" + "=" * 60)
        print("           MODELING CHECKLIST SUMMARY")
        print("=" * 60)
        
        # Iterate through each item and display with appropriate icon
        for item in self.items:
            # Determine which emoji to use based on status
            if self.status[item] == "done":
                icon = "✅"  # Green checkmark for success
            elif self.status[item] == "fail":
                icon = "❌"  # Red X for failure
            else:
                icon = "⏳"  # Hourglass for pending/waiting
            
            # Print formatted line: [ICON]  Item description
            print(f"{icon}  {item}")
        
        # Print footer border
        print("=" * 60 + "\n")


# =============================================================================
# SELF-TEST: Run this module directly to test the ChecklistLogger
# =============================================================================
if __name__ == "__main__":
    """
    When this file is run directly (not imported), execute a simple test
    to verify the ChecklistLogger works correctly.
    """
    print("🧪 Testing ChecklistLogger...\n")
    
    # Create a new logger instance
    logger = ChecklistLogger()
    
    # Mark a few items as done to demonstrate functionality
    logger.mark_done("Finance Dataset Downloaded & Verified")
    logger.mark_done("Data Scaled with MinMaxScaler")
    
    # Mark one item as failed to show error handling
    logger.mark_fail("Model Trained with Early Stopping")
    
    # Display the complete summary
    logger.print_summary()
    
    # Show raw status data structure
    print("Raw status output:", logger.get_status())
