package ihm;

import javax.swing.JPanel;
import javax.swing.JLabel;
import java.awt.Dimension;

public final class Chart {

    private JPanel img;
    private int height, width;

    public Chart(int width, int height) {
        this.height = height;
        this.width = width;
        draw();
    }

    public void update(int actualGeneration, double melhor, double media, double pior) {
        // Update the placeholder label with latest values
        if (img.getComponentCount() > 0 && img.getComponent(0) instanceof JLabel) {
            ((JLabel) img.getComponent(0)).setText(
                String.format("Gen:%d Best:%.2f Avg:%.2f Worst:%.2f", actualGeneration, melhor, media, pior)
            );
        }
    }

    public void clear() {
        if (img.getComponentCount() > 0 && img.getComponent(0) instanceof JLabel) {
            ((JLabel) img.getComponent(0)).setText("No data");
        }
    }

    public JPanel getImage() {
        return img;
    }

    private void draw() {
        img = new JPanel();
        JLabel label = new JLabel("Chart placeholder");
        img.add(label);
        img.setPreferredSize(new Dimension(width - 20, height - 20));
        img.setMinimumSize(new Dimension(width - 20, height - 20));
        img.setMaximumSize(new Dimension(width - 20, height - 20));
    }
}