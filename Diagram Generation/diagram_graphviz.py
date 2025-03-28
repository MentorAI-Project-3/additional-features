from graphviz import Digraph

def create_deep_learning_overview():
    dot = Digraph()
    dot.attr(rankdir='TB')
    
    # Main Topic
    dot.node('DL', 'Deep Learning', shape='box', style='filled', fillcolor='lightblue')
    
    # Key Components
    dot.node('NN', 'Neural Networks', shape='ellipse', style='filled', fillcolor='lightgreen')
    dot.edge('DL', 'NN')
    
    # Layers in Deep Learning
    dot.node('Layers', 'Layers in Deep Learning', shape='box', style='filled', fillcolor='lightyellow')
    dot.edge('NN', 'Layers')
    dot.node('Input', 'Input Layer', shape='ellipse')
    dot.node('Hidden', 'Hidden Layers', shape='ellipse')
    dot.node('Output', 'Output Layer', shape='ellipse')
    dot.edge('Layers', 'Input')
    dot.edge('Layers', 'Hidden')
    dot.edge('Layers', 'Output')
    
    # Activation Functions
    dot.node('AF', 'Activation Functions', shape='box', style='filled', fillcolor='lightcoral')
    dot.edge('NN', 'AF')
    dot.node('ReLU', 'ReLU')
    dot.node('Sigmoid', 'Sigmoid')
    dot.node('Tanh', 'Tanh')
    dot.edge('AF', 'ReLU')
    dot.edge('AF', 'Sigmoid')
    dot.edge('AF', 'Tanh')
    
    # Backpropagation & Optimization
    dot.node('BO', 'Backpropagation & Optimization', shape='box', style='filled', fillcolor='lightgray')
    dot.edge('NN', 'BO')
    dot.node('GD', 'Gradient Descent')
    dot.node('Optimizers', 'Optimizers (Adam, SGD)')
    dot.edge('BO', 'GD')
    dot.edge('BO', 'Optimizers')
    
    # Applications
    dot.node('Applications', 'Applications of Deep Learning', shape='box', style='filled', fillcolor='lightpink')
    dot.edge('DL', 'Applications')
    dot.node('CV', 'Computer Vision')
    dot.node('NLP', 'Natural Language Processing')
    dot.node('Auto', 'Autonomous Systems')
    dot.node('Finance', 'Finance & Healthcare')
    dot.edge('Applications', 'CV')
    dot.edge('Applications', 'NLP')
    dot.edge('Applications', 'Auto')
    dot.edge('Applications', 'Finance')
    
    return dot

# Generate and render the graph
dot = create_deep_learning_overview()
dot.render('chatbot_diagram', format='png', cleanup=True)