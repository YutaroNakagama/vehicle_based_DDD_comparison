# Create a vertically oriented pipeline diagram with "Data \n Processing" formatting
import matplotlib.pyplot as plt

# Define the pipeline components with desired formatting
components = [
    "Vehicle Dynamics Data",  # Input data without a box
#    "DataPreprocessing",
    "Feature Extraction",
    "Feature Selection",
    "Classification",
    "Driver State (Drowsy/Awake)"  # Output data without a box
]

# Create the plot
fig, ax = plt.subplots(figsize=(6, 3))

# Adjust vertical spacing between components
y_positions = [len(components) - i - 1 for i in range(len(components))]

# Add arrows between the components
for i in range(len(y_positions) - 1):
    ax.annotate("", xy=(0.5, y_positions[i + 1] + 0.2), xytext=(0.5, y_positions[i] - 0.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8, shrinkA=2, shrinkB=2))

# Add rightward branching arrow from Classification to Feature Selection
classification_y = y_positions[2]  # Classification position
feature_selection_y = y_positions[3]  # Feature Selection position
# Create a branch that goes rightward from the middle of the downward arrow
ax.annotate("", xy=(.705, classification_y), xytext=(.61, classification_y),
            arrowprops=dict(arrowstyle="<-", color="k", lw=0.8, linestyle="dashed"))
# Arrow from the rightward branch to Feature Selection
ax.annotate("", xy=(.7, feature_selection_y - .15), xytext=(.7, classification_y + .05),
            arrowprops=dict(arrowstyle="-", color="k", lw=0.8, linestyle="dashed"))
ax.annotate("", xy=(.705, feature_selection_y - .1), xytext=(.5, feature_selection_y - .1),
            arrowprops=dict(arrowstyle="-", color="k", lw=0.8, linestyle="dashed"))

#ax.text(.691, (feature_selection_y + classification_y) / 2, "  wrapper method", 
#        fontsize=12, ha='left', va='center', fontname='Times New Roman') 

# Plot each component as a box or text
for y, component in zip(y_positions, components):
    if "Vehicle" in component or "Awake" in component:
        ax.text(0.5, y, component, fontsize=14, ha='center', va='center',
                fontname='Times New Roman')  # No bounding box for Input/Output
    else:
        ax.text(0.5, y, component, fontsize=14, ha='center', va='center',
                fontname='Times New Roman',
                bbox=dict(boxstyle="square,pad=0.3", edgecolor="black", facecolor="white"))

# Set axis limits and hide axes
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, len(components) - 0.5)
ax.axis("off")

# Show the diagram
plt.tight_layout()
plt.savefig("flowchart.pdf", format="pdf", bbox_inches="tight")
plt.show()

