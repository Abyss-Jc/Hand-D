import unittest
import torch
from visualizer_app.gesture_engine import GESTURE_LABELS, _GestureMLP

class TestGestureEngine(unittest.TestCase):

    def test_gesture_labels_sync(self):
        """
        [Calidad]: Asegura que el motor esté sincronizado con el .pth de 4 clases actual.
        """
        # Volvemos a las 4 clases originales
        expected_labels = ['Fist', 'Index_Finger', 'Ruler', 'Thumb_Up']
        self.assertEqual(GESTURE_LABELS, expected_labels, 
                         "Las etiquetas deben ser 4 para coincidir con el checkpoint actual.")

    def test_model_architecture_output(self):
        """
        [Calidad]: Verifica que la red devuelva 4 clases.
        """
        model = _GestureMLP()
        dummy_input = torch.randn(1, 69)
        output = model(dummy_input)
        
        self.assertEqual(output.shape, (1, 4), 
                         "El modelo debe devolver 4 clases para no crashear con el .pth local.")

if __name__ == '__main__':
    unittest.main()
