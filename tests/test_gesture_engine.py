import unittest
import torch
from visualizer_app.gesture_engine import GESTURE_LABELS, _GestureMLP

class TestGestureEngine(unittest.TestCase):

    def test_gesture_labels_sync(self):
        """
        [Calidad - Prevención de Bugs]: Asegura que nuestro motor visualizador 
        esté 100% sincronizado con las 5 clases que Josué entrenó en gesture_classifier.py.
        """
        expected_labels = ['Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle']
        self.assertEqual(GESTURE_LABELS, expected_labels, 
                         "Las etiquetas de los gestos no coinciden con el modelo de entrenamiento.")

    def test_model_architecture_output(self):
        """
        [Calidad - Robustez]: Verifica que la red neuronal inicializada para inferencia 
        pueda manejar el estado 'Idle' (5 neuronas de salida en lugar de 4).
        """
        model = _GestureMLP()
        # Creamos un tensor simulado (dummy) de 1 sample x 69 features
        dummy_input = torch.randn(1, 69)
        output = model(dummy_input)
        
        # El output debe tener shape [1, 5]
        self.assertEqual(output.shape, (1, 5), 
                         "La arquitectura del modelo debe devolver 5 clases (incluyendo Idle).")

if __name__ == '__main__':
    unittest.main()
