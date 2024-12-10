import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

graph_theme = "seaborn-v0_8-deep"
color_map = "Set1"
plt.style.use(graph_theme)


def moving_average(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calcula el promedio móvil de un vector x con una ventana de tamaño window_size.
    """
    return np.array(
        [
            np.mean(x[i : i + window_size])
            for i in range(len(x) - window_size + 1)
        ]
    )


def plot_segment(
    segment_signal: np.ndarray,
    sr: int,
    segment_index: int,
    segment_duration: float,
    audio_path: str,
    window_size: int,
    output_dir: str,
):
    """
    Grafica un segmento individual siguiendo la estética de plot_by_directory,
    mostrando dos subplots:
    - Amplitud vs Tiempo (en escala lineal, similar a Audacity)
    - Amplitud vs Frecuencia (en dB, con línea cruda y media móvil)

    Además, coloca los ticks de frecuencia cada 1000 Hz y maneja segmentos incompletos.
    """
    # Calcular tiempos de inicio y fin del segmento en segundos
    start_time = (segment_index - 1) * segment_duration
    end_time = segment_index * segment_duration

    # Si el segmento está vacío o no es suficientemente largo, se ignora
    if len(segment_signal) == 0:
        print(f"Segmento {segment_index} vacío, se omite.")
        return

    # Eje de tiempo para el segmento
    time = np.linspace(start_time, end_time, len(segment_signal))

    # Normalizar la amplitud a su máximo para mostrar entre -1 y 1
    max_val = np.max(np.abs(segment_signal))
    if max_val == 0:
        print(f"Segmento {segment_index} es silencio puro, se omite.")
        return
    normalized_signal = segment_signal / max_val

    # FFT y cálculo de magnitud en dB en el dominio de frecuencia
    fft_result = np.fft.fft(segment_signal)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sr)
    magnitude = np.abs(fft_result)

    positive_frequencies = frequencies[: len(frequencies) // 2]
    positive_magnitude_db = 20 * np.log10(
        magnitude[: len(magnitude) // 2] + 1e-9
    )

    # Seleccionamos un color base
    color = plt.get_cmap(color_map)(0.0)

    # Crear figura
    plt.figure(figsize=(10, 6))

    # Subplot 1: Amplitud vs Tiempo (lineal)
    plt.subplot(2, 1, 1)
    plt.plot(
        time,
        normalized_signal,
        color=color,
        linewidth=0.8,
        alpha=0.8,
        zorder=2,
    )
    plt.title("Amplitud vs Tiempo (Lineal)")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud (Escala lineal)")
    plt.grid(True, zorder=1)
    plt.ylim(
        -1, 1
    )  # Ajuste de eje para mostrar claramente la forma de onda normalizada

    # Subplot 2: Amplitud vs Frecuencia (dB)
    plt.subplot(2, 1, 2)
    # Línea cruda con mayor opacidad (alpha=0.2)
    plt.plot(
        positive_frequencies,
        positive_magnitude_db,
        color=color,
        linewidth=0.1,
        alpha=0.2,
        zorder=2,
    )

    # Media móvil
    smoothed_magnitude_db = moving_average(positive_magnitude_db, window_size)
    plt.plot(
        positive_frequencies[: len(smoothed_magnitude_db)],
        smoothed_magnitude_db,
        label="Media móvil",
        color=color,
        linewidth=1.5,
        zorder=4,
    )

    plt.title("Amplitud vs Frecuencia (dB)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud (dB)")
    plt.grid(True, zorder=1)

    if len(positive_frequencies) > 0:
        plt.xlim(0, None)
        max_freq = positive_frequencies[-1]
        # Ticks cada 1000 Hz
        xticks = np.arange(0, max_freq, 1000)
        plt.xticks(xticks, rotation=45)

    plt.legend(loc="upper right", fontsize="small")

    # Anotación con toda la info
    annotation_text = (
        f"Archivo: {os.path.basename(audio_path)}\n"
        f"Segmento: {segment_index}\n"
        f"Rango tiempo: {start_time:.2f}-{end_time:.2f} s\n"
        f"Duración segmento: {segment_duration} s"
    )
    plt.annotate(
        annotation_text,
        xy=(0.02, 0.05),
        xycoords="axes fraction",
        ha="left",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"
        ),
    )

    plt.tight_layout()

    # Guardar figura
    output_file = os.path.join(output_dir, f"segment_{segment_index}.png")
    plt.savefig(output_file, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Segmenta un audio y grafica amplitud-tiempo (lineal) y amplitud-frecuencia por cada segmento con estilo plot_by_directory."
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Ruta al archivo de audio",
    )
    parser.add_argument(
        "--segment_duration",
        type=float,
        required=True,
        help="Duración de cada segmento (s)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Tamaño de la ventana para el promedio móvil",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="graphs",
        help="Directorio de salida para las figuras de cada segmento",
    )
    args = parser.parse_args()

    audio_path = args.audio_path
    segment_duration = args.segment_duration
    window_size = args.window_size
    output_dir = args.output_dir

    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)
    num_segments = int(np.floor(total_duration / segment_duration))

    if num_segments == 0:
        print(f"El audio es muy corto para segmentar {segment_duration} s.")
        return

    print(f"Audio cargado: {audio_path}")
    print(f"Duración total: {total_duration:.2f} s, SR: {sr} Hz")
    print(f"Creando {num_segments} segmento(s) de {segment_duration} s")

    samples_per_segment = int(segment_duration * sr)

    for i in range(1, num_segments + 1):
        start_sample = int((i - 1) * samples_per_segment)
        end_sample = int(i * samples_per_segment)

        if end_sample > len(y):
            end_sample = len(y)

        segment_signal = y[start_sample:end_sample]

        # Verificar longitud del segmento
        if len(segment_signal) < samples_per_segment:
            print(f"Segmento {i} demasiado corto, se omite.")
            continue

        plot_segment(
            segment_signal=segment_signal,
            sr=sr,
            segment_index=i,
            segment_duration=segment_duration,
            audio_path=audio_path,
            window_size=window_size,
            output_dir=output_dir,
        )

    print(f"Figuras guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
