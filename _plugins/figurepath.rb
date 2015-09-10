class FigurePathTag < Liquid::Tag

  def initialize(tag_name, markup, tokens)
    super
  end

  def render(context)
    here = File.dirname(context.registers[:page]["path"])
    relative = context.registers[:site].config.fetch("absolute_figures",false)

    if relative
      return File.join(here,"figures")+"/"
    else
      return "figures/"
    end

  end
end

Liquid::Template.register_tag('figurepath', FigurePathTag)
