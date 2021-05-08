import logging
import functools
try:
    import comet_ml
    comet_installed = True
except:
    comet_installed = False


class CometLogger:
    def __init__(self, comet=False, **kwargs):
        global comet_installed
        self._logging = None
        self._comet_args = kwargs
        if comet == False:
            self._logging = False
        elif comet == True and comet_installed == False:
            raise Exception("Comet not installed. Run 'pip install comet-ml'")
    
    def _requiresComet(method):
        @functools.wraps(method)
        def wrapper(*args, **kwargs):
            self = args[0]
            global comet_installed
            if self._logging is None and comet_installed:
                self._logging = False
                try:
                    comet_ml.init()
                    if comet_ml.get_global_experiment() is not None:
                        logging.warning("You have already created a comet \
                                        experiment manually, which might \
                                        cause clashes")
                    self._experiment = comet_ml.Experiment(**self._comet_args)
                    self._logging = True
                except Exception as e:
                    logging.warning(e)

            if self._logging == True:
                return method(*args, **kwargs)
        return wrapper
    
    @_requiresComet
    def end(self):
        """Ends an experiment."""
        self._experiment.end()
        comet_ml.config.experiment = None

    @_requiresComet
    def log_others(self, dictionary):
        """Reports dictionary of key/values to the Other tab on Comet.ml.
        Useful for reporting datasets attributes, datasets path, unique identifiers etc.

        Args:
            dictionary:  dict of key/values where value is Any type
              of value (str,int,float..)
        """
        self._experiment.log_others(dictionary)

    @_requiresComet
    def log_metric(self, name, value, step=None, epoch=None,
                   include_context=True):
        """Logs a general metric (i.e accuracy, f1)..

        Args:
            name: String - Name of your metric
            value: Float/Integer/Boolean/String
            step: Optional. Used as the X axis when plotting on comet.ml
            epoch: Optional. Used as the X axis when plotting on comet.ml
            include_context: Optional. If set to True (the default),
                the current context will be logged along the metric.
        """
        self._experiment.log_metric(name, value, step, epoch,
                                    include_context)

    @_requiresComet
    def log_code(self, file_name=None, folder=None, code=None,
                 code_name=None):
        """Log additional source code files.
        Args:   
            file_name: optional, string: the file path of the file to log
            folder: optional, string: the folder path where the code files
                are stored
            code: optional, string: source code, either as a string or a
                file-like object (like StringIO). If passed, code_name is mandatory
            code_name: optional, string: name of the source code file when
                code parameter is passed
        """
        self._experiment.log_code(file_name, folder, code, code_name)

    @_requiresComet
    def add_tag(self, tag):
        """Add a tag to the experiment.
        Args:   
            tag: String. A tag to add to the experiment.
        """
        self._experiment.add_tag(tag)

    @_requiresComet
    def set_epoch(self, epoch):
        """Sets the current epoch in the training process.
        Args:   
            epoch: Integer value.
        """
        self._experiment.set_epoch(epoch)

    @_requiresComet
    def context_manager(self, context):
        """A context manager to mark the beginning and the end of
           the training phase. 
        Args:   
            context: String.
        Returns:
            context_manager
        """
        return self._experiment.context_manager(context)

    @_requiresComet
    def log_audio(self, audio_data, sample_rate=None, file_name=None,
                  metadata=None, overwrite=False, copy_to_tmp=True,
                  step=None):
        """Logs the audio Asset determined by audio data.
        Args:     
        audio_data: String or a numpy array - either the file path
            of the file you want to log, or a numpy array given to
            scipy.io.wavfile.write for wav conversion.
        sample_rate: Integer - Optional. The sampling rate given to
            scipy.io.wavfile.write for creating the wav file.
        file_name: String - Optional. A custom file name to be displayed.
            If not provided, the filename from the audio_data argument
            will be used.
        metadata: Some additional data to attach to the the audio asset.
            Must be a JSON-encodable dict.
        overwrite: if True will overwrite all existing assets with the same name.
        copy_to_tmp: If audio_data is a numpy array, then this flag
            determines if the WAV file is first copied to a temporary
            file before upload. If copy_to_tmp is False, then it is sent
            directly to the cloud.
        step: Optional. Used to associate the audio asset to a specific step.
        """
        self._experiment.log_audio(audio_data, sample_rate, file_name,
                                   metadata, overwrite, copy_to_tmp,
                                   step)
        