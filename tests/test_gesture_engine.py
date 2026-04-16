import unittest
import torch
from visualizer_app.gesture_engine import GESTURE_LABELS, _GestureMLP

class TestGestureEngine(unittest.TestCase):

    def test_gesture_labels_sync(self):
        """
        [Calidad]: Asegura que el motor esté sincronizado con el modelo de 5 clases.
        """
        expected_labels = ['Fist', 'Index_Finger', 'Ruler', 'Thumb_Up', 'Idle']
        self.assertEqual(GESTURE_LABELS, expected_labels, 
                         "Las etiquetas deben ser 5 para coincidir con el modelo entrenado.")

    def test_model_architecture_output(self):
        """
        [Calidad]: Verifica que la red devuelva 5 clases.
        """
        model = _GestureMLP()
        dummy_input = torch.randn(1, 69)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 5), 
                         "El modelo debe devolver 5 clases.")

if __name__ == '__main__':
    unittest.main()
