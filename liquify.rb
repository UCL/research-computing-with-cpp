inpath = ARGV[0]

require 'rubygems'

require 'liquid'
require_relative '_plugins/idio'
require_relative '_plugins/figurepath'
require 'ostruct'

# Liquid::Template.error_mode = :strict Removed for older liquid compatibility

@template = Liquid::Template.parse(File.read(inpath))

if ARGV[1] == "rel"
  absolute_figures = false
else
  absolute_figures = true
end

site = OpenStruct.new
site.config={
  "idio" => {
    "fence" => true
  },
  "absolute_figures" => absolute_figures
}

page = {
  "path" => inpath
}

registers = {
  :site => site,
  :page => page
}

context = Liquid::Context.new({}, {}, registers)

print @template.render(context)
